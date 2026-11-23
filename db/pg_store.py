"""Generic PostgreSQL store.

Subclasses bind ``row_model`` and ``annotation_fields``. This base owns the
connection, batch insert, metadata ranking and the segment lookup used to
back-fill annotations for vector-only candidates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, case, create_engine, desc, literal, or_, select, text
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from db.models import Base
from db.relational_base import RelationalStoreBase

logger = logging.getLogger(__name__)

#: Columns never used as a filter (surrogate key / bookkeeping).
FILTER_DENYLIST: Tuple[str, ...] = ("id", "processing_ts", "created_at")


class PostgresRelationalStore(RelationalStoreBase):
    #: ORM row model for this dataset's own table. Set by subclasses.
    row_model: type
    #: Annotation fields this dataset may rank / filter. Set by subclasses.
    annotation_fields: Tuple[str, ...] = ()

    def __init__(self) -> None:
        self.engine = create_engine(settings.pg_dsn, pool_pre_ping=True)
        self._session = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def init(self) -> None:
        Base.metadata.create_all(self.engine, tables=[self.row_model.__table__])
        logger.info("Table '%s' ready", self.row_model.__table__.name)

    def drop(self) -> None:
        table = self.row_model.__table__
        Base.metadata.drop_all(self.engine, tables=[table])

        seq_name = f"{table.name}_id_seq"
        with self.engine.begin() as conn:
            quoted = conn.dialect.identifier_preparer.quote(seq_name)
            conn.execute(text(f"DROP SEQUENCE IF EXISTS {quoted};"))
        logger.warning("Table '%s' dropped (sequence %s)", table.name, seq_name)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def _known_columns(self) -> set[str]:
        return {column.name for column in self.row_model.__table__.columns}

    def _row_values(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Map a pipeline record onto this table's columns.

        Top-level keys win; ``metadata`` / ``annotation`` only fill columns that
        are still unset. Unknown keys are dropped, so a single pipeline can feed
        both datasets without per-dataset row builders.
        """
        known = self._known_columns()

        values = {key: val for key, val in item.items() if key in known}
        for source in (item.get("metadata"), item.get("annotation")):
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                if key in known and values.get(key) is None:
                    values[key] = value
        return values

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0

        objects = [self.row_model(**self._row_values(row)) for row in rows]
        try:
            with self._session() as session:
                session.add_all(objects)
                session.commit()
        except Exception:
            logger.exception("Relational batch insert failed (%d rows)", len(objects))
            return 0

        logger.info(
            "Inserted %d rows into '%s'", len(objects), self.row_model.__table__.name
        )
        return len(objects)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def _columns(self) -> Dict[str, Any]:
        return {column.name: column for column in self.row_model.__table__.columns}

    def _filter_pairs(self, filters: Optional[Dict[str, Any]]):
        columns = self._columns()
        return [
            (key, value)
            for key, value in (filters or {}).items()
            if key in columns and key not in FILTER_DENYLIST
        ]

    def search_metadata(
        self,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rank rows by annotation-match count.

        * annotation conditions are OR-ed (score = matched / condition count);
        * metadata filters are AND-ed and act as a hard filter;
        * with neither, the call returns nothing: an unscored full-table scan
          would inject arbitrary candidates and make recall non-reproducible.
        """
        columns = self._columns()
        ann = [
            (key, value)
            for key, value in (annotation_conditions or {}).items()
            if key in self.annotation_fields and key in columns
        ]
        meta = self._filter_pairs(metadata_filters)
        if metadata_filters and not meta:
            logger.warning(
                "None of the metadata filters %s exist on '%s'; relational recall skipped",
                sorted(metadata_filters),
                self.row_model.__table__.name,
            )

        if not ann and not meta:
            return []

        if ann:
            score_expressions = [case((columns[key] == value, 1), else_=0) for key, value in ann]
            match_count = score_expressions[0]
            for expression in score_expressions[1:]:
                match_count = match_count + expression
        else:
            match_count = literal(1)

        clauses = []
        if ann:
            clauses.append(or_(*[columns[key] == value for key, value in ann]))
        if meta:
            clauses.append(and_(*[columns[key] == value for key, value in meta]))

        stmt = select(self.row_model, match_count.label("match_count")).where(*clauses)
        # Deterministic ranking: score first, then stable column tie-breakers.
        stmt = stmt.order_by(
            desc(match_count),
            columns["clip_id"].asc(),
            columns["segment_start"].asc(),
            columns["id"].asc(),
        ).limit(limit)

        try:
            with self._session() as session:
                rows_out = session.execute(stmt).all()
        except Exception:
            logger.exception("Relational metadata query failed")
            return []

        n_conditions = len(ann)
        results: List[Dict[str, Any]] = []
        for row, matched in rows_out:
            hit = {name: getattr(row, name) for name in columns}
            hit["score"] = float(matched) / n_conditions if n_conditions else 1.0
            results.append(hit)
        return results

    def get_segments_by_ids(self, segment_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full rows for ``segment_ids``; used to back-fill annotations."""
        if not segment_ids:
            return []

        columns = self._columns()
        stmt = select(self.row_model).where(
            columns["segment_id"].in_(list(segment_ids))
        )
        try:
            with self._session() as session:
                rows = session.execute(stmt).scalars().all()
        except Exception:
            logger.exception("Relational segment lookup failed")
            return []

        return [{name: getattr(row, name) for name in columns} for row in rows]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.engine.dispose()
