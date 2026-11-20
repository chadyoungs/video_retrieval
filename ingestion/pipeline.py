from __future__ import annotations

import gc
import logging
import os
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from annotators.factory import get_default_annotator
from db.store_factory import get_relational_store, get_vector_store
from encoders.factory import get_default_encoder
from encoders.video_loader import read_video_frames_raw
from ingestion.segmenter import segment_bounds
from utils.dataset_adapter import load_video_entries
from config.settings import settings

logger = logging.getLogger(__name__)

class IngestionPipeline:
    def __init__(
        self,
        dataset_name: str,
        drop_sql: bool = False,
        drop_milvus: bool = False,
        limit: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.limit = limit

        self.rel_store = get_relational_store(dataset_name=dataset_name)
        self.vec_store = get_vector_store(dataset_name=dataset_name)

        if drop_sql:
            logger.info("Drop SQL table for %s", self.dataset_name)
            self.rel_store.drop()
        if drop_milvus:
            logger.info("Drop Milvus collection for %s", self.dataset_name)
            self.vec_store.drop()

        self.rel_store.init()
        self.vec_store.init()

        self.encoder = get_default_encoder()
        self.annotator = get_default_annotator()

        # buffer for db batch insert, to reduce frequent I/O
        self._sql_buffer: list[dict[str, Any]] = []
        self._milvus_buffer: list[dict[str, Any]] = []
        self._buffer_cap = settings.batch_size_db 

        self.stats = {
            "total_videos": 0,
            "processed_videos": 0,
            "failed_videos": 0,
            "total_segments": 0,
            "skipped_segments": 0,
            "inserted_segments": 0,
        }

    def _flush_buffer(self) -> None:
        if not self._sql_buffer and not self._milvus_buffer:
            return
        try:
            sql_len = len(self._sql_buffer)
            milvus_len = len(self._milvus_buffer)

            if sql_len > 0:
                self.rel_store.batch_insert(self._sql_buffer)
            if milvus_len > 0:
                self.vec_store.batch_insert(self._milvus_buffer)
            
            # clear buffer
            self._sql_buffer.clear()
            self._milvus_buffer.clear()

        except Exception as e:
            logger.exception("Buffer flush failed, segments discarded")
            self.stats["skipped_segments"] += sql_len

    def _push_row(self, sql_row: dict | None, milvus_row: dict | None) -> None:
        if sql_row is not None:
            self._sql_buffer.append(sql_row)

        if milvus_row is not None:
            self._milvus_buffer.append(milvus_row)

        if len(self._sql_buffer) >= self._buffer_cap or len(self._milvus_buffer) >= self._buffer_cap:
            self._flush_buffer()

    def _process_single_video(self, video_entry: dict[str, Any]) -> None:
        video_file = video_entry["video_file"]
        raw_meta = video_entry.get("metadata", {})
        entry_dataset: str = raw_meta.get("dataset_name", self.dataset_name)
        try:
            seg_starts, seg_ends = segment_bounds(video_file, settings.clip_duration_sec, settings.clip_overlap_sec)
            if not seg_starts:
                logger.warning("No segments generated for %s", video_file)
                return

            for s, e in zip(seg_starts, seg_ends):
                self.stats["total_segments"] += 1
                frames = read_video_frames_raw(
                    video_path=video_file,
                    start=s,
                    end=e,
                    sample_rate=settings.frame_sampling_rate,
                )
                if not frames:
                    logger.warning("Segment [%.2f,%.2f] no frames, skip", s, e)
                    self.stats["skipped_segments"] += 1
                    continue

                # annotation
                anno = self.annotator.annotate(
                    video_path=video_file,
                    start=s,
                    end=e,
                    sample_rate=settings.frame_sampling_rate,
                    preloaded_frames=frames,
                )

                # embedding into vector
                emb = self.encoder.encode(
                    video_path=video_file,
                    start=s,
                    end=e,
                    preloaded_frames=frames,
                )

                if emb is None or anno is None:
                    logger.warning("Segment [%.2f,%.2f] emb/anno is None, skip", s, e)
                    self.stats["skipped_segments"] += 1
                    continue
                
                video_file_name = os.path.basename(video_file)
                video_file_path_dir = os.path.dirname(video_file)
                clip_id = Path(video_file_name).stem
                segment_id = f"{entry_dataset}_{clip_id}_{int(s * 1000)}_{int(e * 1000)}"
               
                sql_row = {
                    "segment_id": segment_id,
                    "video_file_name": video_file_name,
                    "video_file_path": video_file_path_dir,
                    "segment_start": s,
                    "segment_end": e,
                    "clip_id": clip_id,
                    "dataset_name": entry_dataset,
                    "dataset_version": raw_meta.get("dataset_version"),
                    "dataset_split": raw_meta.get("dataset_split"),
                    "annotation_model": settings.annotation_model_version,
                    "processing_ts": datetime.now(timezone.utc).isoformat(),
                    **anno,
                }

                # milvus data
                milvus_row = {
                    "segment_id": segment_id,
                    "video_file_name": video_file_name,
                    "video_file_path": video_file_path_dir,
                    "segment_start": s,
                    "segment_end": e,
                    "dataset_name": entry_dataset,
                    "clip_id": clip_id,
                    "embedding_model": settings.embedding_model_version,
                    "clip_vector": emb,
                }
                
                self._push_row(sql_row, milvus_row)
                self.stats["inserted_segments"] += 1

            self.stats["processed_videos"] += 1

        except Exception as e:
            logger.exception("Process video failed: %s", video_file)
            self.stats["failed_videos"] += 1

    def run(self):
        video_entries = load_video_entries()
        if self.limit is not None:
            video_entries = video_entries[:self.limit]
        self.stats["total_videos"] = len(video_entries)

        with tqdm(total=len(video_entries), desc="Ingestion") as pbar:
            for entry in video_entries:
                self._process_single_video(entry)
                pbar.update(1)
        
        self._flush_buffer()

        logger.info("Ingest finished. stats: %s", self.stats)
