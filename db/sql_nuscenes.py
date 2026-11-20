"""NuScenes relational store: its own table, tokens, timing and ego motion."""

from __future__ import annotations

from typing import Tuple

from db.models import NuScenesSegmentRow
from db.pg_store import PostgresRelationalStore


class NuScenesRelationalStore(PostgresRelationalStore):
    row_model = NuScenesSegmentRow
    annotation_fields: Tuple[str, ...] = (
        "driving_context",
        "weather",
        "people_count",
        "lighting",
        "time_of_day",
        "cyclist_presence",
        "road_user_density",
        "traffic_flow",
        "semantic_summary",
    )
