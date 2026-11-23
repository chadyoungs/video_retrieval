"""Factory: get relational / vector store by dataset_key.

Implementations are imported inside the factory functions so that importing any
retrieval module does not require pymilvus / SQLAlchemy to be installed.
"""

from __future__ import annotations

from typing import Literal

from db.relational_base import RelationalStoreBase
from db.vector_base import VectorStoreBase

DatasetName = Literal["activitynet", "nuscenes"]

_SUPPORTED: tuple[str, ...] = ("activitynet", "nuscenes")


def _check(dataset_name: str) -> None:
    if dataset_name not in _SUPPORTED:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name!r}; expected one of {_SUPPORTED}"
        )


def get_relational_store(dataset_name: DatasetName) -> RelationalStoreBase:
    _check(dataset_name)
    if dataset_name == "activitynet":
        from db.sql_activitynet import ActivityNetRelationalStore

        return ActivityNetRelationalStore()

    from db.sql_nuscenes import NuScenesRelationalStore

    return NuScenesRelationalStore()


def get_vector_store(dataset_name: DatasetName) -> VectorStoreBase:
    _check(dataset_name)
    from db.milvus_store import MilvusVectorStore

    return MilvusVectorStore(dataset_name=dataset_name)
