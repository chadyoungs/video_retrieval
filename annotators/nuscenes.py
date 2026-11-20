"""NuScenes annotator: full driving-context annotation set."""

from __future__ import annotations

from annotators.base import BaseAnnotator
from annotators.rules import (
    ANNOTATION_RULES_NUSCENES,
    DEFAULT_ANNOTATION_NUSCENES,
)


class NuScenesAnnotator(BaseAnnotator):
    """Annotator for NuScenes segment clips."""

    rules = ANNOTATION_RULES_NUSCENES
    default_annotation = DEFAULT_ANNOTATION_NUSCENES

    def dataset_name(self) -> str:
        return "nuscenes"
