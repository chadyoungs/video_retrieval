"""Annotator factory: return the implementation for a dataset key."""

from __future__ import annotations

from typing import Literal, Optional

from src.annotators.activitynet import ActivityNetAnnotator
from src.annotators.base import BaseAnnotator
from src.annotators.nuscenes import NuScenesAnnotator
from src.config.settings import settings

DatasetKey = Literal["activitynet", "nuscenes"]


def get_annotator(dataset_key: DatasetKey) -> BaseAnnotator:
    """Return the annotator for ``dataset_key``."""
    if dataset_key == "activitynet":
        return ActivityNetAnnotator()
    if dataset_key == "nuscenes":
        return NuScenesAnnotator()
    raise ValueError(f"Unsupported dataset_key: {dataset_key!r}")


def get_default_annotator() -> BaseAnnotator:
    """Annotator matching ``settings.dataset_source``."""
    return get_annotator(settings.dataset_source)
