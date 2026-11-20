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
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(String(128), nullable=False)
    video_file_name = Column(String(64))
    video_file_path = Column(String(128))
    segment_start = Column(Float)
    segment_end = Column(Float)
    clip_id = Column(String(64))
    dataset_name = Column(String(64))
    dataset_version = Column(String(64))
    dataset_split = Column(String(64))
    annotation_model = Column(String(128))
    processing_ts = Column(String(64))
    quality_flag = Column(String(16), default="ok")

    # Scene annotation subset for ActivityNet.
    scene_env = Column(String(32))
    scene_type = Column(String(32))
    people_count = Column(String(32))
    lighting = Column(String(32))
    time_of_day = Column(String(32))
    
    semantic_summary = Column(String(512))


class NuScenesSegmentRow(Base):
    """NuScenes-only table."""

    __tablename__ = "nuscenes_video_segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    segment_id = Column(String(128), nullable=False)
    video_file_name = Column(String(256))
    video_file_path = Column(String(512))
    segment_start = Column(Float)
    segment_end = Column(Float)
    clip_id = Column(String(128))
    dataset_name = Column(String(64))
    dataset_version = Column(String(64))
    dataset_split = Column(String(64))
    camera_channel = Column(String(64), default="unknown")
    annotation_model = Column(String(256))
    processing_ts = Column(String(64))
    quality_flag = Column(String(16), default="ok")

    # Scene.
    location = Column(String(128))
    
    # Timing and source-frame statistics.
    clip_fps = Column(Float)
    clip_frame_count = Column(Integer)
    clip_info = Column(String(256))

    # Driving-context annotation fields.
    driving_context = Column(String(128))
    weather = Column(String(32))
    people_count = Column(String(32))
    lighting = Column(String(32))
    time_of_day = Column(String(32))
    cyclist_presence = Column(String(32))
    road_user_density = Column(String(32))
    traffic_flow = Column(String(32))

    # Ego motion summary.
    ego_avg_speed_mps = Column(Float)
    ego_max_speed_mps = Column(Float)
    
    semantic_summary = Column(String(512))
