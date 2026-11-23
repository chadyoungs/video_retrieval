"""Legacy dataset discovery helper.

The ingestion pipeline now runs through ``datasets.factory`` + the typed
``VideoEntry`` adapters. This module is kept only for old scripts; prefer
``datasets.factory.get_dataset_adapter(...).discover()``.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from config.settings import settings

logger = logging.getLogger(__name__)

ALLOWED_DATASET_SOURCES = {"activitynet", "nuscenes"}


def _build_activitynet_entries() -> List[Dict]:
    data_root = Path(settings.activitynet_data_root)
    if not data_root.exists():
        logger.warning("ActivityNet root dir does not exist: %s", data_root)
        return []

    pattern = f"*.{settings.activitynet_file_format}"
    entries = []
    for p in data_root.glob(pattern):
        entries.append({
            "video_file": str(p),
            "metadata": {
                "dataset_name": "activitynet",
                "dataset_version": "unknown",
                "dataset_split": "unknown",
                "camera_channel": "unknown",
            },
        })
    logger.info("Loaded %d activitynet video entries from %s", len(entries), data_root)
    return entries


def _build_nuscenes_entries() -> List[Dict]:
    mapping_file = Path(settings.nuscenes_clip_mapping_file)
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"nuScenes clip mapping file not found: {mapping_file}"
        )

    with mapping_file.open("r", encoding="utf-8") as f:
        clip_mapping = json.load(f)

    if not isinstance(clip_mapping, list):
        raise TypeError("clip_mapping json must be a list")

    video_root = Path(settings.nuscenes_video_root)
    entries: List[Dict] = []
    seen_paths = set()

    for item in clip_mapping:
        clip_rel_path = item.get("clip_relative_path") or item.get("clip_file_name")
        if not clip_rel_path:
            logger.debug("Skip mapping item without clip path: %s", item)
            continue

        clip_path = video_root / clip_rel_path
        clip_path_str = str(clip_path)
        if clip_path_str in seen_paths:
            continue
        if not clip_path.exists():
            logger.debug("Clip file missing: %s", clip_path)
            continue

        metadata = {
            "dataset_name": settings.nuscenes_dataset_name,
            "dataset_version": item.get("dataset_version", settings.nuscenes_dataset_version),
            "dataset_split": item.get("dataset_split", settings.nuscenes_dataset_split),
            "camera_channel": item.get("camera_channel", "CAM_FRONT"),
            "clip_id": item.get("clip_id"),
            "scene_token": item.get("scene_token"),
            "sample_start_token": item.get("sample_start_token"),
            "sample_end_token": item.get("sample_end_token"),
            "log_token": item.get("log_token"),
            "location": item.get("location"),
            "map_token": item.get("map_token"),
            "clip_start_ts": item.get("clip_start_ts"),
            "clip_end_ts": item.get("clip_end_ts"),
            "clip_fps": item.get("fps"),
            "clip_frame_count": item.get("frame_count"),
            "ego_avg_speed_mps": item.get("ego_avg_speed_mps"),
            "ego_max_speed_mps": item.get("ego_max_speed_mps"),
            "quality_flag": item.get("quality_flag", "ok"),
        }
        entries.append({
            "video_file": clip_path_str,
            "metadata": metadata
        })
        seen_paths.add(clip_path_str)

    logger.info("Loaded %d nuscenes video entries from mapping %s", len(entries), mapping_file)
    return entries


def load_video_entries() -> List[Dict]:
    if settings.dataset_source not in ALLOWED_DATASET_SOURCES:
        raise ValueError(
            f"Unsupported settings.dataset_source={settings.dataset_source}. "
            f"Expected one of {sorted(ALLOWED_DATASET_SOURCES)}"
        )

    if settings.dataset_source == "nuscenes":
        return _build_nuscenes_entries()
    return _build_activitynet_entries()
