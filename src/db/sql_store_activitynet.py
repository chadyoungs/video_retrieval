"""ActivityNet relational store: simplified scene annotation subset."""

from __future__ import annotations

from typing import Tuple

from app.db.pg_store import PostgresRelationalStore

ACTIVITYNET_ANNOTATION_FIELDS: Tuple[str, ...] = (
    "scene_env",
    "scene_type",
    "weather",
    "lighting",
    "time_of_day",
)


class ActivityNetRelationalStore(PostgresRelationalStore):
    """Metadata store for ActivityNet segments.

    ActivityNet annotates the simplified scene subset only; the NuScenes-only
    driving fields stay null on this dataset's rows.
    """

    annotation_fields = ACTIVITYNET_ANNOTATION_FIELDS
