"""NuScenes relational store: full driving-context annotation set."""

from __future__ import annotations

from typing import Tuple

from app.db.pg_store import PostgresRelationalStore

NUSCENES_ANNOTATION_FIELDS: Tuple[str, ...] = (
    "scene_env",
    "scene_type",
    "weather",
    "lighting",
    "time_of_day",
    "person_count",
    "driving_context",
    "road_user_density",
    "traffic_flow",
)


class NuScenesRelationalStore(PostgresRelationalStore):
    """Metadata store for NuScenes segments.

    Keeps the full driving-context annotation set alongside the shared scene
    fields; scoring and record shape are inherited from the base store.
    """

    annotation_fields = NUSCENES_ANNOTATION_FIELDS
