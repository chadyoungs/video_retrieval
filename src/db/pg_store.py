"""Generic PostgreSQL store.

Subclasses only bind ``row_model`` and ``annotation_fields``. The shared base
handles connection, batch insert, metadata ranking and record mapping. Only
columns that exist on this dataset's own table are written; unknown metadata
or annotation keys are dropped.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    and_,
    case,
    create_engine,
    desc,
    literal,
    or_,
    select,
)
from sqlalchemy.orm import sessionmaker

from src.config.settings import settings
from src.db.models import Base
from src.db.relational_base import RelationalStoreBase

logger = logging.getLogger(__name__)


# Metadata filter fields shared by both datasets. Fields missing on a
# dataset's own table are ignored at search time.
METADATA_FILTER_FIELDS: Tuple[str, ...] = (
    "dataset_name",
    "dataset_version",
    "dataset_split",
    "camera_channel",
    "clip_id",
    "scene_token",
    "location",
)


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
        # create_all / drop_all are SQLAlchemy MetaData methods; ``tables=[...]``
        # limits the operation to this store's own table only.
        Base.metadata.create_all(
            self.engine, tables=[self.row_model.__table__]
        )
        logger.info("Table '%s' ready", self.row_model.__table__.name)

    def drop(self) -> None:
        Base.metadata.drop_all(
            self.engine, tables=[self.row_model.__table__]
        )
        logger.warning("Table '%s' dropped", self.row_model.__table__.name)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def _known_columns(self) -> set[str]:
        return {column.name for column in self.row_model.__table__.columns}

    def _row_values(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Map a record onto this table's columns; unknown keys are dropped."""
        annotation = item.get("annotation") or {}
        metadata = item.get("metadata") or {}
        known = self._known_columns()

        values: Dict[str, Any] = {
            "video_file_name": item.get("video_file_name"),
            "video_file_path": item.get("video_file_path"),
            "segment_start": item.get("segment_start"),
            "segment_end": item.get("segment_end"),
        }

        # Provenance and annotation values land on matching columns only.
        for source in (metadata, annotation):
            for key, value in source.items():
                if key in known:
                    values[key] = value

        for provenance in (
            "embedding_model",
            "annotation_model",
            "processing_ts",
        ):
            if provenance in known:
                values[provenance] = metadata.get(provenance)

        return {key: value for key, value in values.items() if key in known}

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0

        objects = [self.row_model(**self._row_values(row)) for row in rows]
        try:
            with self._session() as session:
                session.add_all(objects)
                session.commit()
            logger.info(
                "Inserted %d rows into '%s'",
                len(objects),
                self.row_model.__table__.name,
            )
            return len(objects)
        except Exception:
            logger.exception("Relational batch insert failed")
            return 0

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def search_metadata(
        self,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Ranked metadata search.

        * annotation conditions are OR-ed together;
        * metadata filters are AND-ed together;
        * score = matched annotation fields / condition count, or 1.0 when
          there are no annotation conditions.
        """
        ann = [
            (key, value)
            for key, value in (annotation_conditions or {}).items()
            if key in self.annotation_fields
        ]
        columns = {
            column.name: column
            for column in self.row_model.__table__.columns
        }
        meta = [
            (key, value)
            for key, value in (metadata_filters or {}).items()
            if key in METADATA_FILTER_FIELDS and key in columns
        ]

        if ann:
            score_cases = [
                case((columns[key] == value, 1), else_=0)
                for key, value in ann
            ]
            match_count = score_cases[0]
            for score_case in score_cases[1:]:
                match_count = match_count + score_case
        else:
            match_count = literal(1)

        clauses = []
        if ann:
            clauses.append(or_(*[columns[key] == value for key, value in ann]))
        if meta:
            clauses.append(and_(*[columns[key] == value for key, value in meta]))

        stmt = select(
            self.row_model, match_count.label("match_count")
        )
        if clauses:
            stmt = stmt.where(*clauses)
        stmt = stmt.order_by(desc(match_count)).limit(limit)

        try:
            with self._session() as session:
                rows_out = session.execute(stmt).all()
        except Exception:
            logger.exception("Relational query failed")
            return []

        n_conditions = len(ann)
        results: List[Dict[str, Any]] = []
        for row, match_count in rows_out:
            hit = {
                column.name: getattr(row, column.name)
                for column in self.row_model.__table__.columns
            }
            hit["score"] = (
                float(match_count) / n_conditions
                if n_conditions > 0
                else 1.0
            )
            results.append(hit)
        return results

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.engine.dispose()
