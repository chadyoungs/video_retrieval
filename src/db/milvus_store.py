"""Milvus vector store, implements VectorStoreBase
Accept collection name externally, so activitynet / nuscenes use separate collections
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional
from pymilvus import DataType, MilvusClient
from src.db.vector_base import VectorStoreBase
from src.config.settings import settings

logger = logging.getLogger(__name__)
_ALLOWED_FILTER_FIELDS = {
    "dataset_name",
    "camera_channel",
    "clip_id",
    "scene_token",
    "embedding_model",
}
_VALUE_PATTERN = re.compile(r"^[\w\-.:/ ]+$")


class MilvusVectorStore(VectorStoreBase):
    def __init__(self, collection_name: str) -> None:
        self._client: Optional[MilvusClient] = None
        self.collection_name = collection_name
        self.alias = getattr(settings, "milvus_alias", "default")
        self.dim = settings.embedding_dim
        self.metric_type = settings.milvus_metric_type

    def _connect(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(
                uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
                alias=self.alias,
            )
            logger.info("Connected to Milvus %s:%s", settings.milvus_host, settings.milvus_port)
        return self._client

    def create_collection(self, drop_existing: bool = False) -> None:
        client = self._connect()
        if client.has_collection(self.collection_name, using=self.alias):
            if drop_existing:
                logger.warning("Drop milvus collection %s", self.collection_name)
                client.drop_collection(self.collection_name, using=self.alias)
            else:
                logger.info("Collection %s exists, skip create", self.collection_name)
                return
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True, description="Video CLIP Embeddings")
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("video_file_name", DataType.VARCHAR, max_length=256)
        schema.add_field("video_file_path", DataType.VARCHAR, max_length=512)
        schema.add_field("segment_start", DataType.FLOAT)
        schema.add_field("segment_end", DataType.FLOAT)
        schema.add_field("dataset_name", DataType.VARCHAR, max_length=64)
        schema.add_field("camera_channel", DataType.VARCHAR, max_length=64)
        schema.add_field("clip_id", DataType.VARCHAR, max_length=128)
        schema.add_field("scene_token", DataType.VARCHAR, max_length=128)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=256)
        schema.add_field("clip_vector", DataType.FLOAT_VECTOR, dim=self.dim)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="clip_vector",
            index_type="HNSW",
            metric_type=self.metric_type,
            params={"M": 16, "efConstruction": 200},
        )
        client.create_collection(self.collection_name, schema=schema, index_params=index_params)
        logger.info("Milvus collection %s created", self.collection_name)

    def insert_vectors(self, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not records:
            return None
        return self._connect().insert(collection_name=self.collection_name, data=records)

    @staticmethod
    def _build_filter_expression(metadata_filters: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata_filters:
            return None
        clauses = []
        for k, v in metadata_filters.items():
            if k not in _ALLOWED_FILTER_FIELDS:
                continue
            if not _VALUE_PATTERN.match(str(v)):
                continue
            safe_v = str(v).replace("\\", "\\\\").replace("'", "\\'")
            clauses.append(f"{k} == '{safe_v}'")
        return " and ".join(clauses) if clauses else None

    def search_vectors(
        self,
        query_embedding: List[float],
        limit: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        client = self._connect()
        filter_expr = self._build_filter_expression(metadata_filters)
        try:
            res = client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=limit,
                output_fields=[
                    "video_file_name", "video_file_path",
                    "segment_start", "segment_end",
                    "dataset_name", "camera_channel",
                    "clip_id", "scene_token", "embedding_model",
                ],
                search_params={"metric_type": self.metric_type, "params": {"ef": 64}},
                filter=filter_expr,
            )
        except Exception as e:
            logger.error("Milvus search err: %s", e)
            return []
        hits = []
        for hit in res[0]:
            entity = hit.get("entity", {})
            dist = float(hit.get("distance", 0.0))
            score = (dist + 1.0) / 2.0
            hits.append({
                "video_file_name": entity.get("video_file_name"),
                "video_file_path": entity.get("video_file_path"),
                "segment_start": entity.get("segment_start"),
                "segment_end": entity.get("segment_end"),
                "dataset_name": entity.get("dataset_name"),
                "camera_channel": entity.get("camera_channel"),
                "clip_id": entity.get("clip_id"),
                "scene_token": entity.get("scene_token"),
                "embedding_model": entity.get("embedding_model"),
                "distance": dist,
                "score": score,
            })
        return hits

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
