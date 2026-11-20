"""Annotator factory: return the implementation for a dataset key."""

from __future__ import annotations

from typing import Literal

from annotators.activitynet import ActivityNetAnnotator
from annotators.base import BaseAnnotator
from annotators.nuscenes import NuScenesAnnotator
from config.settings import settings

DatasetName = Literal["activitynet", "nuscenes"]


def get_annotator(dataset_name: DatasetName) -> BaseAnnotator:
    """Return the annotator for ``dataset_name``."""
    if dataset_name == "activitynet":
        return ActivityNetAnnotator()
    if dataset_name == "nuscenes":
        return NuScenesAnnotator()
    raise ValueError(f"Unsupported dataset_name: {dataset_name!r}")


def get_default_annotator() -> BaseAnnotator:
    """Annotator matching ``settings.dataset_source``."""
    return get_annotator(settings.dataset_source)
