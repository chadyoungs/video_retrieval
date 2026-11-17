"""Factory: get relational / vector store by dataset_key."""

from __future__ import annotations

from typing import Literal

from src.db.milvus_store import MilvusVectorStore
from src.db.relational_base import RelationalStoreBase
from src.db.sql_activitynet import ActivityNetRelationalStore
from src.db.sql_nuscenes import NuScenesRelationalStore
from src.db.vector_base import VectorStoreBase

DatasetKey = Literal["activitynet", "nuscenes"]


def get_relational_store(
    dataset_key: DatasetKey,
) -> RelationalStoreBase:
    if dataset_key == "activitynet":
        return ActivityNetRelationalStore()
    if dataset_key == "nuscenes":
        return NuScenesRelationalStore()
    raise ValueError(f"Unknown dataset_key: {dataset_key}")


def get_vector_store(dataset_key: DatasetKey) -> VectorStoreBase:
    return MilvusVectorStore(dataset_key=dataset_key)
