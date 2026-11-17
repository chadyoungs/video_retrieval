"""Adapter factory: return the dataset adapter for a dataset key."""

from __future__ import annotations

from typing import Literal

from src.config.settings import settings
from src.datasets.activitynet import ActivityNetAdapter
from src.datasets.base import DatasetAdapterBase
from src.datasets.nuscenes import NuScenesAdapter

DatasetKey = Literal["activitynet", "nuscenes"]


def get_dataset_adapter(dataset_key: DatasetKey) -> DatasetAdapterBase:
    """Return the adapter for ``dataset_key``; default to settings.source."""
    if dataset_key == "activitynet":
        return ActivityNetAdapter()
    if dataset_key == "nuscenes":
        return NuScenesAdapter()
    raise ValueError(f"Unsupported dataset_key: {dataset_key!r}")


def get_default_adapter() -> DatasetAdapterBase:
    """Adapter matching ``settings.dataset_source``."""
    return get_dataset_adapter(settings.dataset_source)
