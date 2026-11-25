"""NuScenes adapter: read the builder clip mapping and resolve segment videos.

The builder mapping records segment timing, scene linkage and ego motion; this 
adapter resolves file paths, skips missing clips and renames builder fields to 
the store metadata field names.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config.settings import settings
from datasets.base import DatasetAdapterBase, VideoEntry

logger = logging.getLogger(__name__)


class NuScenesAdapter(DatasetAdapterBase):
    """Discover NuScenes segments listed in the clip mapping JSON.

    Unlike ActivityNet's continuous video, nuScenes ships 2 Hz keyframes
    (two images cover one second) and the builder already cuts each scene
    window into its own mp4: one file = one segment. Re-windowing with the
    ActivityNet sliding window or resampling at 10 fps would be wrong, so
    this adapter declares ``presegmented`` and the native keyframe rate.
    """

    presegmented = True

    @property
    def frame_sampling_rate(self) -> float:
        # Native keyframe rate: every frame of the clip is kept.
        return float(settings.builder_output_fps)

    def discover(self) -> List[VideoEntry]:
        mapping_path = Path(settings.nuscenes_clip_mapping_file)
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"nuScenes clip mapping file not found: {mapping_path}"
            )

        with mapping_path.open("r", encoding="utf-8") as f:
            clip_mapping: List[Dict[str, Any]] = json.load(f)

        video_root = Path(settings.nuscenes_video_root)
        entries: List[VideoEntry] = []

        for item in clip_mapping:
            rel_path = (
                item.get("clip_relative_path") or item.get("clip_file_name")
            )
            if not rel_path:
                continue

            clip_path = video_root / rel_path
            if not clip_path.exists():
                logger.warning("Skipping missing NuScenes clip: %s", clip_path)
                continue

            metadata: Dict[str, Any] = {
                "dataset_name": item.get(
                    "dataset_name", settings.dataset_source
                ),
                "dataset_version": item.get(
                    "dataset_version", settings.nuscenes_dataset_version
                ),
                "dataset_split": item.get(
                    "dataset_split", settings.nuscenes_dataset_split
                ),
                "camera_channel": item.get(
                    "camera_channel", settings.nuscenes_camera_channel
                ),
                "clip_id": item.get("clip_id"),
                "segment_id": item.get("segment_id"),
                "location": item.get("location"),
                "clip_start_ts": item.get("clip_start_ts"),
                "clip_end_ts": item.get("clip_end_ts"),
                # Builder fields -> store metadata fields.
                "clip_fps": item.get("fps"),
                "clip_frame_count": item.get("frame_count"),
                "ego_avg_speed_mps": item.get("ego_avg_speed_mps"),
                "ego_max_speed_mps": item.get("ego_max_speed_mps"),
                "quality_flag": item.get("quality_flag", "ok"),
            }

            # segment_start/end in the mapping are scene-relative timestamps,
            # NOT offsets inside the clip file (the file starts at frame 0).
            # They stay informational; ingestion decodes the whole clip file.
            entries.append(
                VideoEntry(
                    video_file_name=clip_path.name,
                    video_file_path=str(clip_path),
                    segment_start=item.get("segment_start"),
                    segment_end=item.get("segment_end"),
                    metadata=metadata,
                )
            )

        logger.info("Discovered %d NuScenes segments", len(entries))
        return entries
