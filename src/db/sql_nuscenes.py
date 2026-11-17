"""NuScenes relational store: its own table, tokens, timing and ego motion."""

from __future__ import annotations

from typing import Tuple

from src.db.models import NuScenesSegmentRow
from src.db.pg_store import PostgresRelationalStore


class NuScenesRelationalStore(PostgresRelationalStore):
    row_model = NuScenesSegmentRow
    annotation_fields: Tuple[str, ...] = (
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
