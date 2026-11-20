"""Generate overlapping sliding-window CAM_FRONT clips for nuScenes.

The legacy record shape, mapping JSON fields and ego-motion statistics are 
preserved; the main change is that onescene is cut into fixed-duration 
segments with configurable overlap, and the mapping list is later handed 
to the relational / vector stores as ``SegmentRecord`` metadata.

Segment boundaries follow source timestamps (microseconds), so irregular
sample rates are handled correctly. Overlap is expressed in seconds:

    step_sec = clip_duration_sec - clip_overlap_sec

A trailing segment shorter than ``min_clip_sec`` is dropped, except the very
first window of a scene.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class _FrameEntry:
    """One CAM_FRONT sample_data frame inside a scene."""

    sample_token: str
    sd_token: str
    timestamp: int  # microseconds
    image_path: Path
    sample: Dict[str, Any]
    sample_data: Dict[str, Any]


class NuScenesVideoBuilder:
    """Cut nuScenes scenes into overlapping segment videos and write mapping."""

    def __init__(
        self,
        dataroot: str,
        meta_root: str,
        output_dir: str,
        dataset_version: str,
        dataset_split: str,
        camera_channel: str,
        output_fps: float,
        segment_duration_sec: float,
        segment_overlap_sec: float,
        min_segment_sec: float,
    ) -> None:
        self.dataroot = Path(dataroot)
        self.meta_root = Path(meta_root)
        self.output_dir = Path(output_dir)
        self.dataset_version = dataset_version
        self.dataset_split = dataset_split
        self.camera_channel = camera_channel
        self.output_fps = output_fps

        self.duration_us = segment_duration_sec * 1e6
        # Guard: overlap must stay below the window duration; cap at 90%.
        overlap_sec = min(segment_overlap_sec, segment_duration_sec * 0.9)
        self.overlap_us = overlap_sec * 1e6
        self.step_us = max(self.duration_us - self.overlap_us, 1.0)
        self.min_segment_us = min_segment_sec * 1e6

        self._samples: Dict[str, Dict[str, Any]] = {}
        self._sample_data: Dict[str, Dict[str, Any]] = {}
        self._logs: Dict[str, Dict[str, Any]] = {}
        self._ego_poses: Dict[str, Dict[str, Any]] = {}

    # ------------------------- metadata loading ------------------------- #

    @staticmethod
    def _load_table(meta_root: Path, name: str) -> List[Dict[str, Any]]:
        table_file = meta_root / f"{name}.json"
        with table_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_tables(self) -> None:
        """Index nuScenes metadata tables by token; tolerate missing ego_pose."""
        self._samples = {
            x["token"]: x for x in self._load_table(self.meta_root, "sample")
        }
        self._sample_data = {
            x["token"]: x for x in self._load_table(self.meta_root, "sample_data")
        }
        self._logs = {x["token"]: x for x in self._load_table(self.meta_root, "log")}
        try:
            ego_pose = self._load_table(self.meta_root, "ego_pose")
        except FileNotFoundError:
            logger.warning("ego_pose.json missing; ego speed fields will be null.")
            ego_pose = []
        self._ego_poses = {x["token"]: x for x in ego_pose}

    # ------------------------- scene frame chain ------------------------ #

    def _scene_frames(self, scene: Dict[str, Any]) -> List[_FrameEntry]:
        """Walk the scene sample chain and collect camera-channel frames."""
        frames: List[_FrameEntry] = []
        seen: set[str] = set()
        cursor = scene.get("first_sample_token", "")
        while cursor:
            sample = self._samples.get(cursor)
            if not sample:
                break
            sd_token = sample.get("data", {}).get(self.camera_channel)
            sd = self._sample_data.get(sd_token) if sd_token else None
            if sd and sd_token not in seen:
                seen.add(sd_token)
                frames.append(
                    _FrameEntry(
                        sample_token=sample.get("token", ""),
                        sd_token=sd_token,
                        timestamp=int(sd.get("timestamp", 0) or 0),
                        image_path=self.dataroot / sd["filename"],
                        sample=sample,
                        sample_data=sd,
                    )
                )
            cursor = sample.get("next", "")

        # Drop frames without a usable timestamp, then order by time.
        frames = [f for f in frames if f.timestamp > 0]
        frames.sort(key=lambda f: f.timestamp)
        return frames

    # ------------------------- ego motion ------------------------------ #

    def _ego_motion(self, frames: List[_FrameEntry]) -> Dict[str, Optional[float]]:
        """Per-segment ego speed summary; matches the legacy statistic shape."""
        speeds: List[float] = []
        for prev, curr in zip(frames, frames[1:]):
            prev_pose = self._ego_poses.get(prev.sample_data.get("ego_pose_token"))
            curr_pose = self._ego_poses.get(curr.sample_data.get("ego_pose_token"))
            if not prev_pose or not curr_pose:
                continue
            prev_t = prev_pose.get("translation", [0, 0, 0])
            curr_t = curr_pose.get("translation", [0, 0, 0])
            distance = sum((curr_t[i] - prev_t[i]) ** 2 for i in range(3)) ** 0.5
            dt = (curr.timestamp - prev.timestamp) / 1e6
            if dt > 0:
                speeds.append(distance / dt)

        if not speeds:
            return {"ego_avg_speed_mps": None, "ego_max_speed_mps": None}
        return {
            "ego_avg_speed_mps": sum(speeds) / len(speeds),
            "ego_max_speed_mps": max(speeds),
        }

    # ------------------------- segmentation ---------------------------- #

    def _segment_windows(
        self, frames: List[_FrameEntry]
    ) -> List[List[_FrameEntry]]:
        """Split the ordered frame list into timestamp-based overlapping windows."""
        t0 = frames[0].timestamp
        last = frames[-1].timestamp
        windows: List[List[_FrameEntry]] = []

        start = t0
        seg_idx = 0
        while start < last:
            end = min(start + self.duration_us, last)
            window = [f for f in frames if start <= f.timestamp <= end]
            if window:
                span = window[-1].timestamp - window[0].timestamp
                # Keep the first window even when it is shorter than the floor;
                # drop every other short trailing segment.
                if span >= self.min_segment_us or seg_idx == 0:
                    windows.append(window)
            seg_idx += 1
            start = t0 + seg_idx * self.step_us
        return windows

    # ------------------------- video writing --------------------------- #

    @staticmethod
    def _read_image(path: Path) -> Optional[Any]:
        return cv2.imread(str(path))

    def _write_segment(
        self,
        frames: List[_FrameEntry],
        clip_path: Path,
        width: int,
        height: int,
    ) -> int:
        """Write one segment video; return number of frames actually written."""
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.output_fps,
            (width, height),
        )
        written = 0
        try:
            for frame in frames:
                image = self._read_image(frame.image_path)
                if image is None:
                    logger.warning("Skipping unreadable frame: %s", frame.image_path)
                    continue
                if image.shape[1] != width or image.shape[0] != height:
                    image = cv2.resize(image, (width, height))
                writer.write(image)
                written += 1
        finally:
            writer.release()

        if written == 0 and clip_path.exists():
            clip_path.unlink()
        return written

    # ------------------------- main build ------------------------------ #

    def build(
        self, scene_tokens: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Build segments for all selected scenes; return the mapping list."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.load_tables()

        scenes = self._load_table(self.meta_root, "scene")
        if scene_tokens:
            wanted = set(scene_tokens)
            scenes = [s for s in scenes if s.get("token") in wanted]

        mapping: List[Dict[str, Any]] = []
        for scene in scenes:
            try:
                mapping.extend(self._build_scene(scene))
            except Exception as exc:  # isolate one bad scene, keep the rest
                logger.error("Scene %s failed: %s", scene.get("token"), exc)
        return mapping

    def _build_scene(self, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
        scene_token = scene.get("token")
        scene_name = scene.get("name", scene_token)
        log_token = scene.get("log_token")
        log_rec = self._logs.get(log_token, {})
        location = log_rec.get("location")
        map_token = log_rec.get("map_token")

        frames = self._scene_frames(scene)
        if not frames:
            return []

        # Reference size comes from the first readable frame of the scene.
        ref_image: Optional[Any] = None
        for frame in frames:
            ref_image = self._read_image(frame.image_path)
            if ref_image is not None:
                break
        if ref_image is None:
            logger.warning("No readable image for scene %s", scene_token)
            return []
        height, width = ref_image.shape[:2]

        records: List[Dict[str, Any]] = []
        windows = self._segment_windows(frames)
        t0 = frames[0].timestamp

        for seg_idx, window in enumerate(windows):
            clip_id = f"{scene_name}_{scene_token[:8]}_seg{seg_idx:03d}"
            clip_file_name = f"{clip_id}.mp4"
            clip_path = self.output_dir / clip_file_name

            try:
                written = self._write_segment(window, clip_path, width, height)
            except Exception as exc:  # isolate one bad segment
                logger.error("Segment %s failed: %s", clip_id, exc)
                if clip_path.exists():
                    clip_path.unlink()
                continue
            if written == 0:
                continue

            first, last = window[0], window[-1]
            ego = self._ego_motion(window)
            records.append(
                {
                    "dataset_name": settings.nuscenes_dataset_name,
                    "dataset_version": self.dataset_version,
                    "dataset_split": self.dataset_split,
                    "camera_channel": self.camera_channel,
                    "clip_id": clip_id,
                    "clip_file_name": clip_file_name,
                    "clip_relative_path": clip_file_name,
                    "segment_index": seg_idx,
                    "scene_token": scene_token,
                    "log_token": log_token,
                    "location": location,
                    "map_token": map_token,
                    "sample_start_token": first.sample_token,
                    "sample_end_token": last.sample_token,
                    "clip_start_ts": first.timestamp,
                    "clip_end_ts": last.timestamp,
                    "segment_start": round((first.timestamp - t0) / 1e6, 3),
                    "segment_end": round((last.timestamp - t0) / 1e6, 3),
                    "fps": self.output_fps,
                    "frame_count": written,
                    "quality_flag": "ok",
                    **ego,
                }
            )
        return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate overlapping nuScenes CAM_FRONT segment videos"
    )
    parser.add_argument("--dataroot", default=settings.nuscenes_raw_dataroot)
    parser.add_argument("--meta-root", default=settings.nuscenes_meta_root)
    parser.add_argument("--output-dir", default=settings.nuscenes_video_root)
    parser.add_argument(
        "--dataset-version", default=settings.nuscenes_dataset_version
    )
    parser.add_argument("--dataset-split", default=settings.nuscenes_dataset_split)
    parser.add_argument("--camera-channel", default=settings.nuscenes_camera_channel)
    parser.add_argument("--fps", type=float, default=settings.builder_output_fps)
    parser.add_argument("--duration", type=float, default=settings.clip_duration_sec)
    parser.add_argument("--overlap", type=float, default=settings.clip_overlap_sec)
    parser.add_argument(
        "--min-segment-sec", type=float, default=settings.min_clip_sec
    )
    parser.add_argument(
        "--scene-token",
        action="append",
        default=[],
        help="Optional scene token filter (can be repeated)",
    )
    args = parser.parse_args()

    if not args.dataroot or not args.meta_root:
        parser.error(
            "--dataroot and --meta-root are required, either as CLI flags or via "
            "settings.nuscenes_raw_dataroot / settings.nuscenes_meta_root"
        )

    builder = NuScenesVideoBuilder(
        dataroot=args.dataroot,
        meta_root=args.meta_root,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
        dataset_split=args.dataset_split,
        camera_channel=args.camera_channel,
        output_fps=args.fps,
        segment_duration_sec=args.duration,
        segment_overlap_sec=args.overlap,
        min_segment_sec=args.min_segment_sec,
    )
    mapping = builder.build(scene_tokens=args.scene_token or None)

    mapping_path = Path(settings.nuscenes_clip_mapping_file)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(mapping)} {args.camera_channel} clips")
    print(f"Clip mapping written to: {mapping_path}")


if __name__ == "__main__":
    main()
