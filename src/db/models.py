"""ORM models: one PostgreSQL table per dataset.

Alembic setup:
    from src.db.models import Base
    target_metadata = Base.metadata
"""

from __future__ import annotations

from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base. Do NOT replace this with a plain ABC."""


class ActivityNetSegmentRow(Base):
    """ActivityNet-only table."""

    __tablename__ = "activitynet_video_segments"

    segment_id = Column(String(128), primary_key=True)
    video_file_name = Column(String(256), nullable=False)
    video_file_path = Column(String(512), nullable=False)
    segment_start = Column(Float)
    segment_end = Column(Float)
    clip_id = Column(String(128))
    dataset_name = Column(String(64))
    dataset_version = Column(String(64))
    dataset_split = Column(String(64))
    camera_channel = Column(String(64), default="unknown")
    embedding_model = Column(String(256))
    annotation_model = Column(String(256))
    processing_ts = Column(String(64))
    quality_flag = Column(String(16), default="ok")

    # Scene annotation subset for ActivityNet.
    scene_env = Column(String(32))
    scene_type = Column(String(32))
    weather = Column(String(32))
    lighting = Column(String(32))
    time_of_day = Column(String(32))


class NuScenesSegmentRow(Base):
    """NuScenes-only table: tokens, timing, ego motion and driving context."""

    __tablename__ = "nuscenes_video_segments"

    segment_id = Column(String(128), primary_key=True)
    video_file_name = Column(String(256), nullable=False)
    video_file_path = Column(String(512), nullable=False)
    segment_start = Column(Float)
    segment_end = Column(Float)
    clip_id = Column(String(128))
    dataset_name = Column(String(64))
    dataset_version = Column(String(64))
    dataset_split = Column(String(64))
    camera_channel = Column(String(64), default="unknown")
    embedding_model = Column(String(256))
    annotation_model = Column(String(256))
    processing_ts = Column(String(64))
    quality_flag = Column(String(16), default="ok")

    # Scene linkage tokens.
    scene_token = Column(String(128))
    sample_start_token = Column(String(128))
    sample_end_token = Column(String(128))
    log_token = Column(String(128))
    location = Column(String(128))
    map_token = Column(String(128))

    # Timing and source-frame statistics.
    clip_start_ts = Column(Integer)
    clip_end_ts = Column(Integer)
    clip_fps = Column(Float)
    clip_frame_count = Column(Integer)

    # Driving-context annotation fields.
    person_count = Column(String(32))
    driving_context = Column(String(32))
    road_user_density = Column(String(32))
    traffic_flow = Column(String(32))

    # Ego motion summary.
    ego_avg_speed_mps = Column(Float)
    ego_max_speed_mps = Column(Float)
