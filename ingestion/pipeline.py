"""Segment -> annotate -> embed -> persist, one dataset at a time.

The pipeline is deliberately thin: dataset discovery comes from
``datasets.factory``, annotation from ``annotators.factory``, representation
from ``encoders.factory`` and persistence from ``db.store_factory``.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from annotators.factory import get_annotator
from config.settings import settings
from datasets.factory import get_dataset_adapter
from db.store_factory import get_relational_store, get_vector_store
from encoders.factory import get_default_encoder
from encoders.video_loader import read_video_frames_raw
from ingestion.segmenter import segment_bounds

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        dataset_name: str,
        drop_sql: bool = False,
        drop_milvus: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        if dataset_name not in ("activitynet", "nuscenes"):
            raise ValueError(f"Unknown dataset_name: {dataset_name!r}")

        self.dataset_name = dataset_name
        self.limit = limit

        self.rel_store = get_relational_store(dataset_name=dataset_name)
        self.vec_store = get_vector_store(dataset_name=dataset_name)

        if drop_sql:
            logger.info("Dropping SQL table for %s", dataset_name)
            self.rel_store.drop()
        if drop_milvus:
            logger.info("Dropping Milvus collection for %s", dataset_name)
            self.vec_store.drop()

        self.rel_store.init()
        self.vec_store.init()

        self.encoder = get_default_encoder()
        self.annotator = get_annotator(dataset_name)

        # Buffer DB writes: one insert round-trip per batch instead of per segment.
        self._sql_buffer: List[Dict[str, Any]] = []
        self._milvus_buffer: List[Dict[str, Any]] = []
        self._buffer_cap = max(1, settings.batch_size_db)

        self.stats: Dict[str, int] = {
            "total_videos": 0,
            "processed_videos": 0,
            "failed_videos": 0,
            "total_segments": 0,
            "skipped_segments": 0,
            "inserted_segments": 0,
        }

    # ------------------------------------------------------------------ #
    # Buffer writes
    # ------------------------------------------------------------------ #
    def _flush_buffer(self) -> None:
        if not self._sql_buffer and not self._milvus_buffer:
            return

        sql_len = len(self._sql_buffer)
        milvus_len = len(self._milvus_buffer)
        try:
            if sql_len:
                self.rel_store.batch_insert(self._sql_buffer)
            if milvus_len:
                self.vec_store.batch_insert(self._milvus_buffer)
        except Exception:  # noqa: BLE001 - one bad batch must not kill the run
            logger.exception("Buffer flush failed; %d segments discarded", sql_len)
            self.stats["skipped_segments"] += sql_len
        finally:
            self._sql_buffer.clear()
            self._milvus_buffer.clear()

    def _push_row(self, sql_row: Dict[str, Any], milvus_row: Dict[str, Any]) -> None:
        self._sql_buffer.append(sql_row)
        self._milvus_buffer.append(milvus_row)
        if len(self._sql_buffer) >= self._buffer_cap or len(self._milvus_buffer) >= self._buffer_cap:
            self._flush_buffer()

    # ------------------------------------------------------------------ #
    # Per-video work
    # ------------------------------------------------------------------ #
    def _process_single_video(self, video_file: str, raw_meta: Dict[str, Any]) -> None:
        try:
            starts, ends = segment_bounds(
                video_file, settings.clip_duration_sec, settings.clip_overlap_sec
            )
            if not starts:
                logger.warning("No segments generated for %s", video_file)
                return

            video_file_name = os.path.basename(video_file)
            video_dir = os.path.dirname(video_file)
            clip_id = Path(video_file_name).stem

            for start, end in zip(starts, ends):
                self.stats["total_segments"] += 1
                frames = read_video_frames_raw(
                    video_path=video_file,
                    start=start,
                    end=end,
                    sample_rate=settings.frame_sampling_rate,
                )
                if not frames:
                    logger.warning("Segment [%.2f, %.2f] has no frames; skipped", start, end)
                    self.stats["skipped_segments"] += 1
                    continue

                annotation = self.annotator.annotate(
                    video_path=video_file,
                    start=start,
                    end=end,
                    sample_rate=settings.frame_sampling_rate,
                    preloaded_frames=frames,
                )
                embedding = self.encoder.encode(
                    video_path=video_file,
                    start=start,
                    end=end,
                    preloaded_frames=frames,
                )
                if embedding is None or annotation is None:
                    logger.warning("Segment [%.2f, %.2f] missing embedding/annotation", start, end)
                    self.stats["skipped_segments"] += 1
                    continue

                segment_id = f"{self.dataset_name}_{clip_id}_{int(start * 1000)}_{int(end * 1000)}"
                base_row = {
                    "segment_id": segment_id,
                    "video_file_name": video_file_name,
                    "video_file_path": video_dir,
                    "segment_start": float(start),
                    "segment_end": float(end),
                    "clip_id": clip_id,
                    "dataset_name": self.dataset_name,
                    "dataset_version": raw_meta.get("dataset_version", "unknown"),
                    "dataset_split": raw_meta.get("dataset_split", "unknown"),
                    "annotation_model": settings.annotation_model_version,
                    "processing_ts": datetime.now(timezone.utc).isoformat(),
                }

                # Dataset metadata is copied through; the relational store keeps
                # only columns that exist on this dataset's table.
                sql_row = {**raw_meta, **base_row, **annotation}

                milvus_row = {
                    "segment_id": segment_id,
                    "video_file_name": video_file_name,
                    "video_file_path": video_dir,
                    "segment_start": float(start),
                    "segment_end": float(end),
                    "dataset_name": self.dataset_name,
                    "clip_id": clip_id,
                    "camera_channel": raw_meta.get("camera_channel", "unknown"),
                    "embedding_model": settings.embedding_model_version,
                    "clip_vector": [float(x) for x in embedding],
                }

                self._push_row(sql_row, milvus_row)
                self.stats["inserted_segments"] += 1

            self.stats["processed_videos"] += 1

        except Exception:  # noqa: BLE001 - continue with the next video
            logger.exception("Failed to process video: %s", video_file)
            self.stats["failed_videos"] += 1

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def _load_entries(self) -> List[Dict[str, Any]]:
        adapter = get_dataset_adapter(self.dataset_name)
        entries = [
            {"video_file": entry.video_file_path, "metadata": entry.metadata}
            for entry in adapter.discover()
        ]
        if self.limit is not None:
            entries = entries[: self.limit]
        return entries

    def run(self) -> None:
        entries = self._load_entries()
        self.stats["total_videos"] = len(entries)
        logger.info("Ingestion starts: dataset=%s videos=%d", self.dataset_name, len(entries))

        try:
            with tqdm(total=len(entries), desc="Ingestion") as progress:
                for entry in entries:
                    self._process_single_video(entry["video_file"], entry["metadata"])
                    progress.update(1)
        finally:
            self._flush_buffer()

        logger.info("Ingestion finished. stats: %s", self.stats)
