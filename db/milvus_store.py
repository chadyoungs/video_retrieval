"""Milvus vector store: one collection per dataset."""

from __future__ import annotations

import logging
from tokenize import String
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config.settings import settings
from db.vector_base import VectorStoreBase

logger = logging.getLogger(__name__)


# Scalar fields per dataset. ActivityNet collection has no scene_token.
ScalarSpec = Tuple[str, DataType, Optional[int]]

_SCALAR_SCHEMA: Dict[str, List[ScalarSpec]] = {
    "activitynet": [
        ("segment_id", DataType.VARCHAR, 128),
        ("video_file_name", DataType.VARCHAR, 64),
        ("video_file_path", DataType.VARCHAR, 128),
        ("segment_start", DataType.FLOAT, None),
        ("segment_end", DataType.FLOAT, None),
        ("dataset_name", DataType.VARCHAR, 64),
        ("clip_id", DataType.VARCHAR, 64),
        ("embedding_model", DataType.VARCHAR, 128),
    ],
    "nuscenes": [
        ("segment_id", DataType.VARCHAR, 128),
        ("video_file_name", DataType.VARCHAR, 64),
        ("video_file_path", DataType.VARCHAR, 128),
        ("segment_start", DataType.FLOAT, None),
        ("segment_end", DataType.FLOAT, None),
        ("dataset_name", DataType.VARCHAR, 64),
        ("clip_id", DataType.VARCHAR, 128),
        ("embedding_model", DataType.VARCHAR, 128),
    ],
}


class MilvusVectorStore(VectorStoreBase):
    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in _SCALAR_SCHEMA:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        self.dataset_name = dataset_name
        self.collection_name = settings.milvus_collections[dataset_name]
        self.dim = settings.embedding_dim
        self.metric_type = settings.milvus_metric_type
        self._scalars = _SCALAR_SCHEMA[dataset_name]
        self._alias = settings.milvus_alias or "default"
        self._collection: Optional[Collection] = None

        connections.connect(
            alias=self._alias,
            host=settings.milvus_host,
            port=settings.milvus_port,
        )

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _build_schema(self) -> CollectionSchema:
        fields: List[FieldSchema] = []
        
        for name, dtype, max_length in self._scalars:
            if dtype == DataType.VARCHAR and name != "segment_id":
                fields.append(
                    FieldSchema(
                        name=name, dtype=dtype, max_length=max_length
                    )
                )
            elif name == "segment_id":
                fields.append(
                    FieldSchema(
                        name=name, dtype=dtype, max_length=max_length, is_primary=True, description="unique primary key for video segment"
                    )
                )
            else:
                fields.append(FieldSchema(name=name, dtype=dtype))

        fields.append(
            FieldSchema(
                name="clip_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            )
        )
        return CollectionSchema(
            fields, description=f"{self.dataset_name} clip embeddings"
        )

    def init(self) -> None:
        if utility.has_collection(
            self.collection_name, using=self._alias
        ):
            self._collection = Collection(
                self.collection_name, using=self._alias
            )
            self._collection.load()
            return

        self._collection = Collection(
            name=self.collection_name,
            schema=self._build_schema(),
            using=self._alias,
        )
        index_params = {
            "index_type": "HNSW",
            "metric_type": self.metric_type,
            "params": {"M": 16, "efConstruction": 200},
        }
        self._collection.create_index("clip_vector", index_params)
        self._collection.load()
        logger.info(
            "Milvus collection '%s' ready", self.collection_name
        )

    def drop(self) -> None:
        if utility.has_collection(
            self.collection_name, using=self._alias
        ):
            utility.drop_collection(
                self.collection_name, using=self._alias
            )
            logger.warning(
                "Milvus collection '%s' dropped", self.collection_name
            )
        self._collection = None

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def _scalar_value(self, row: Dict[str, Any], name: str, dtype: DataType) -> Any:
        value = row.get(name)
        if value is None:
            return "" if dtype == DataType.VARCHAR else 0.0
        return value

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        if not rows or self._collection is None:
            return 0

        # Milvus expects column-wise lists; vector column goes last.
        columns: List[List[Any]] = []
        for name, dtype, _ in self._scalars:
            columns.append(
                [self._scalar_value(row, name, dtype) for row in rows]
            )
        columns.append([row["clip_vector"] for row in rows])

        self._collection.insert(columns)
        self._collection.flush()
        return len(rows)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def _build_expr(
        self, filters: Optional[Dict[str, Any]]
    ) -> str:
        if not filters:
            return ""

        scalar_names = {name for name, _, _ in self._scalars}
        clauses: List[str] = []
        for key, value in filters.items():
            if key not in scalar_names:
                continue
            if isinstance(value, str):
                escaped = value.replace('"', "'")
                clauses.append(f'{key} == "{escaped}"')
            else:
                clauses.append(f"{key} == {value}")
        return " and ".join(clauses)

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._collection is None:
            return []

        expr = self._build_expr(metadata_filters)
        output_fields = [name for name, _, _ in self._scalars]

        results = self._collection.search(
            data=[query_vector],
            anns_field="clip_vector",
            limit=top_k,
            expr=expr or None,
            output_fields=output_fields,
            search_params={
                "metric_type": self.metric_type,
                "params": {"ef": 64},
            },
        )

        hits: List[Dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.entity
            record = {
                name: entity.get(name) for name in output_fields
            }
            # COSINE distance is [-1, 1] similarity; normalize to [0, 1].
            if self.metric_type == "COSINE":
                record["score"] = (hit.distance + 1.0) / 2.0
            else:
                record["score"] = hit.distance
            record["distance"] = hit.distance
            hits.append(record)
        return hits

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        connections.disconnect(self._alias)
