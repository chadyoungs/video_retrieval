"""ActivityNet relational store: its own table and scene annotation subset."""

from __future__ import annotations

from typing import Tuple

from src.db.models import ActivityNetSegmentRow
from src.db.pg_store import PostgresRelationalStore


class ActivityNetRelationalStore(PostgresRelationalStore):
    row_model = ActivityNetSegmentRow
    annotation_fields: Tuple[str, ...] = (
        "scene_env",
        "scene_type",
        "weather",
        "lighting",
        "time_of_day",
    )
