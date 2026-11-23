"""Milvus vector store: one collection per dataset.

Only scalar fields listed in ``_SCALAR_SCHEMA`` are stored in Milvus; everything
else lives in PostgreSQL. Filters on fields that are not stored here are skipped
with a debug log instead of silently changing recall (PostgreSQL still applies
them, and the retrieval service applies a final hard filter).

``pymilvus`` is imported lazily so the module can be imported on machines that
only need the relational side (tests, tooling).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from db.vector_base import VectorStoreBase

logger = logging.getLogger(__name__)

#: (field name, kind, max_length). ``segment_id`` is the primary key.
_SCALAR_SCHEMA: Dict[str, List[Tuple[str, str, Optional[int]]]] = {
    "activitynet": [
        ("segment_id", "varchar", 256),
        ("video_file_name", "varchar", 256),
        ("video_file_path", "varchar", 512),
        ("segment_start", "float", None),
        ("segment_end", "float", None),
        ("dataset_name", "varchar", 64),
        ("clip_id", "varchar", 128),
        ("embedding_model", "varchar", 128),
    ],
    "nuscenes": [
        ("segment_id", "varchar", 256),
        ("video_file_name", "varchar", 256),
        ("video_file_path", "varchar", 512),
        ("segment_start", "float", None),
        ("segment_end", "float", None),
        ("dataset_name", "varchar", 64),
        ("clip_id", "varchar", 128),
        ("camera_channel", "varchar", 64),
        ("embedding_model", "varchar", 128),
    ],
}

PRIMARY_KEY = "segment_id"


class MilvusVectorStore(VectorStoreBase):
    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in _SCALAR_SCHEMA:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        import pymilvus  # local import: heavy, optional for non-vector tasks

        self._pymilvus = pymilvus
        self.dataset_name = dataset_name
        self.collection_name = settings.milvus_collections[dataset_name]
        self.dim = settings.embedding_dim
        self.metric_type = settings.milvus_metric_type
        self._scalars = _SCALAR_SCHEMA[dataset_name]
        self._max_lengths = {name: length for name, _, length in self._scalars}
        self._alias = settings.milvus_alias or "default"
        self._collection: Optional[Any] = None

        pymilvus.connections.connect(
            alias=self._alias, host=settings.milvus_host, port=settings.milvus_port
        )

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _data_type(self, kind: str):
        if kind == "varchar":
            return self._pymilvus.DataType.VARCHAR
        if kind == "float":
            return self._pymilvus.DataType.FLOAT
        raise ValueError(f"Unsupported scalar kind: {kind}")

    @property
    def scalar_names(self) -> List[str]:
        return [name for name, _, _ in self._scalars]

    def _build_schema(self):
        FieldSchema = self._pymilvus.FieldSchema
        fields = []
        for name, kind, max_length in self._scalars:
            dtype = self._data_type(kind)
            if name == PRIMARY_KEY:
                fields.append(
                    FieldSchema(
                        name=name,
                        dtype=dtype,
                        max_length=max_length,
                        is_primary=True,
                        auto_id=False,
                        description="unique key for one video segment",
                    )
                )
            elif dtype == self._pymilvus.DataType.VARCHAR:
                fields.append(FieldSchema(name=name, dtype=dtype, max_length=max_length))
            else:
                fields.append(FieldSchema(name=name, dtype=dtype))

        fields.append(
            self._pymilvus.FieldSchema(
                name="clip_vector", dtype=self._pymilvus.DataType.FLOAT_VECTOR, dim=self.dim
            )
        )
        return self._pymilvus.CollectionSchema(
            fields, description=f"{self.dataset_name} clip embeddings"
        )

    def init(self) -> None:
        Collection = self._pymilvus.Collection
        if self._pymilvus.utility.has_collection(self.collection_name, using=self._alias):
            self._collection = Collection(self.collection_name, using=self._alias)
            existing_dim = self._vector_dim(self._collection)
            if existing_dim and existing_dim != self.dim:
                raise ValueError(
                    f"Milvus collection '{self.collection_name}' has dim {existing_dim} "
                    f"but settings.embedding_dim is {self.dim} "
                    f"(tier={settings.semantic_model_tier}). Re-ingest with "
                    f"--drop-milvus or switch back to the matching tier."
                )
            self._collection.load()
            logger.info(
                "Milvus collection '%s' loaded (%d entities)",
                self.collection_name,
                self._collection.num_entities,
            )
            return

        self._collection = Collection(
            name=self.collection_name, schema=self._build_schema(), using=self._alias
        )
        index_params = {
            "index_type": "HNSW",
            "metric_type": self.metric_type,
            "params": {"M": 16, "efConstruction": 200},
        }
        self._collection.create_index("clip_vector", index_params)
        self._collection.load()
        logger.info("Milvus collection '%s' created", self.collection_name)

    @staticmethod
    def _vector_dim(collection: Any) -> Optional[int]:
        for field in collection.schema.fields:
            if field.name == "clip_vector":
                return int(field.params.get("dim"))
        return None

    def drop(self) -> None:
        if self._pymilvus.utility.has_collection(self.collection_name, using=self._alias):
            self._pymilvus.utility.drop_collection(self.collection_name, using=self._alias)
            logger.warning("Milvus collection '%s' dropped", self.collection_name)
        self._collection = None

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def _scalar_value(self, row: Dict[str, Any], name: str, kind: str) -> Any:
        value = row.get(name)
        if value is None:
            return "" if kind == "varchar" else 0.0
        if kind == "varchar":
            # Milvus rejects over-long VARCHAR values, so clamp instead of failing
            # the whole batch.
            max_length = self._max_lengths[name]
            text = str(value)
            if max_length and len(text) > max_length:
                logger.warning(
                    "Truncating %s to %d chars for Milvus insert", name, max_length
                )
                return text[:max_length]
            return text
        return float(value)

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        if not rows or self._collection is None:
            return 0

        payload = []
        for row in rows:
            entry = {
                name: self._scalar_value(row, name, kind)
                for name, kind, _ in self._scalars
            }
            entry["clip_vector"] = [float(x) for x in row["clip_vector"]]
            payload.append(entry)

        self._collection.insert(payload)
        self._collection.flush()
        logger.info("Inserted %d vectors into '%s'", len(payload), self.collection_name)
        return len(payload)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def _build_expr(self, filters: Optional[Dict[str, Any]]) -> str:
        if not filters:
            return ""

        clauses: List[str] = []
        for key, value in filters.items():
            if key not in self.scalar_names:
                logger.debug("Milvus cannot filter on %s (not stored); skipped", key)
                continue
            if isinstance(value, (list, tuple, set)):
                rendered = ", ".join(self._literal(v) for v in value)
                clauses.append(f"{key} in [{rendered}]")
            else:
                clauses.append(f"{key} == {self._literal(value)}")
        return " and ".join(clauses)

    @staticmethod
    def _literal(value: Any) -> str:
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._collection is None:
            logger.warning("Milvus collection '%s' not initialised", self.collection_name)
            return []

        expr = self._build_expr(metadata_filters)
        if metadata_filters and not expr:
            logger.debug(
                "None of the requested filters are stored in Milvus; ANN is unfiltered"
            )

        results = self._collection.search(
            data=[[float(x) for x in query_vector]],
            anns_field="clip_vector",
            limit=top_k,
            expr=expr or None,
            output_fields=self.scalar_names,
            param={
                "metric_type": self.metric_type,
                "params": {"ef": max(64, top_k)},
            },
        )

        hits: List[Dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.entity
            record = {name: entity.get(name) for name in self.scalar_names}
            # COSINE distance is similarity in [-1, 1]; shift to [0, 1] so it
            # fuses with the metadata score.
            if self.metric_type == "COSINE":
                record["score"] = (float(hit.distance) + 1.0) / 2.0
            else:
                record["score"] = float(hit.distance)
            record["distance"] = float(hit.distance)
            hits.append(record)
        return hits

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self._pymilvus.connections.disconnect(self._alias)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.exception("Error disconnecting Milvus alias %s", self._alias)
