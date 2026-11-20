"""Factory: get relational / vector store by dataset_key."""

from __future__ import annotations

from typing import Literal

from db.milvus_store import MilvusVectorStore
from db.relational_base import RelationalStoreBase
from db.sql_activitynet import ActivityNetRelationalStore
from db.sql_nuscenes import NuScenesRelationalStore
from db.vector_base import VectorStoreBase

DatasetName = Literal["activitynet", "nuscenes"]


def get_relational_store(
    dataset_name: DatasetName,
) -> RelationalStoreBase:
    if dataset_name == "activitynet":
        return ActivityNetRelationalStore()
    if dataset_name == "nuscenes":
        return NuScenesRelationalStore()
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def get_vector_store(dataset_name: DatasetName) -> VectorStoreBase:
    return MilvusVectorStore(dataset_name=dataset_name)
