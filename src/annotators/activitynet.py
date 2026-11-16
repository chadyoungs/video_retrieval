"""ActivityNet annotator: simplified scene annotation subset."""

from __future__ import annotations

from src.annotators.base import BaseAnnotator
from src.annotators.rules import (
    ANNOTATION_RULES_ACTIVITYNET,
    DEFAULT_ANNOTATION_ACTIVITYNET,
)


class ActivityNetAnnotator(BaseAnnotator):
    """Annotator for ActivityNet whole-video segments."""

    rules = ANNOTATION_RULES_ACTIVITYNET
    default_annotation = DEFAULT_ANNOTATION_ACTIVITYNET

    def dataset_key(self) -> str:
        return "activitynet"
