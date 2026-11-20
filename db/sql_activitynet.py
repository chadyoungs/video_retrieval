"""ActivityNet relational store: its own table and scene annotation subset."""

from __future__ import annotations

from typing import Tuple

from db.models import ActivityNetSegmentRow
from db.pg_store import PostgresRelationalStore


class ActivityNetRelationalStore(PostgresRelationalStore):
    row_model = ActivityNetSegmentRow
    annotation_fields: Tuple[str, ...] = (
        "scene_env",
        "scene_type",
        "people_count",
        "lighting",
        "time_of_day",
        "semantic_summary",
    )
