"""Adapter factory: return the dataset adapter for a dataset key."""

from __future__ import annotations

from typing import Literal

from config.settings import settings
from datasets.activitynet import ActivityNetAdapter
from datasets.base import DatasetAdapterBase
from datasets.nuscenes import NuScenesAdapter

DatasetName = Literal["activitynet", "nuscenes"]


def get_dataset_adapter(dataset_name: DatasetName) -> DatasetAdapterBase:
    """Return the adapter for ``dataset_name``."""
    if dataset_name == "activitynet":
        return ActivityNetAdapter()
    if dataset_name == "nuscenes":
        return NuScenesAdapter()
    raise ValueError(f"Unsupported dataset_name: {dataset_name!r}")


def get_default_adapter() -> DatasetAdapterBase:
    """Adapter matching ``settings.dataset_source``."""
    return get_dataset_adapter(settings.dataset_source)
