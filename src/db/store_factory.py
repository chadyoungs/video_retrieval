"""Store factory: bind the relational and vector stores to one dataset key.

The two stores stay deliberately independent: the relational store knows
nothing about Milvus, and the vector store knows nothing about SQLAlchemy.

* relational store -> per-dataset subclass of :class:`PostgresRelationalStore`;
* vector store    -> one Milvus collection per dataset source.
"""

from __future__ import annotations

from typing import Literal, Optional

from src.config.settings import settings
from src.db.milvus_store import MilvusVectorStore
from src.db.relational_base import RelationalStoreBase
from src.db.sql_store_activitynet import ActivityNetRelationalStore
from src.db.sql_store_nuscenes import NuScenesRelationalStore
from src.db.vector_base import VectorStoreBase

DatasetKey = Literal["activitynet", "nuscenes"]


def get_relational_store(
    dataset_key: Optional[DatasetKey] = None,
) -> RelationalStoreBase:
    """Return the relational store for ``dataset_key`` (default: settings)."""
    key = dataset_key or settings.dataset_source
    if key == "activitynet":
        return ActivityNetRelationalStore()
    if key == "nuscenes":
        return NuScenesRelationalStore()
    raise ValueError(f"Unsupported dataset_key: {key!r}")


def get_vector_store(
    dataset_key: Optional[DatasetKey] = None,
) -> VectorStoreBase:
    """Return the vector store; collection is mapped per dataset source."""
    key = dataset_key or settings.dataset_source
    collection = settings.milvus_collections.get(
        key, settings.milvus_collection
    )
    return MilvusVectorStore(collection_name=collection)
