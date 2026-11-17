from __future__ import annotations

import gc
import logging
import os
from datetime import datetime, timezone
from typing import Any

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from src.annotators.factory import get_default_annotator
from src.db.store_factory import get_relational_store, get_vector_store
from src.encoders.factory import get_default_encoder
from src.encoders.video_loader import read_video_frames_raw
from src.ingestion.segmenter import segment_bounds
from src.utils.dataset_adapter import load_video_entries
from src.config.settings import settings

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
            logger.info("Drop SQL table for dataset %s", dataset_name)
            self.rel_store.drop()
        if drop_milvus:
            logger.info("Drop Milvus collection for dataset %s", dataset_name)
            self.vec_store.drop()

        self.rel_store.init()
        self.vec_store.init()

        self.encoder = get_default_encoder()
        self.annotator = get_default_annotator()

        self.stats = {
            "total_videos": 0,
            "processed_videos": 0,
            "failed_videos": 0,
            "total_segments": 0,
            "skipped_segments": 0,
            "inserted_segments": 0,
        }

    def _flush_batch(self, sql_batch: list[dict[str, Any]], milvus_batch: list[dict[str, Any]]):
        if not sql_batch and not milvus_batch:
            return
        try:
            if sql_batch:
                self.rel_store.batch_insert(sql_batch)
            if milvus_batch:
                self.vec_store.batch_insert(milvus_batch)
            logger.debug("Flushed %d sql rows, %d milvus rows", len(sql_batch), len(milvus_batch))
        except Exception as e:
            logger.exception("Batch insert failed, discard this batch")
            self.stats["skipped_segments"] += len(sql_batch)

    def _process_single_video(self, video_entry: dict[str, Any]) -> tuple[list[dict], list[dict]]:
        """
        Process one video: split to segments, read frames once, annotate + embed,
        return sql_batch, milvus_batch.
        """
        video_file = video_entry["video_file"]
        base_metadata = video_entry.get("metadata", {})
        sql_batch: list[dict] = []
        milvus_batch: list[dict] = []

        try:
            seg_starts, seg_ends = segment_bounds(video_file, settings.clip_duration_sec)
            if not seg_starts:
                return [], []

            segments_raw_frames = []
            for s, e in zip(seg_starts, seg_ends):
                frames = read_video_frames_raw(
                    video_path=video_file,
                    start=s,
                    end=e,
                    sample_rate=settings.frame_sampling_rate,
                )
                segments_raw_frames.append(frames or [])

            # annotate all segments
            annotate_results = []
            for s, e, frames in zip(seg_starts, seg_ends, segments_raw_frames):
                if not frames:
                    annotate_results.append(None)
                    continue
                anno = self.annotator.annotate(
                    video_file=video_file,
                    start=s,
                    end=e,
                    sample_rate=settings.frame_sampling_rate,
                    preloaded_frames=frames,
                )
                annotate_results.append(anno)

            # embed all segments
            embed_results = []
            for s, e, frames in zip(seg_starts, seg_ends, segments_raw_frames):
                if not frames:
                    embed_results.append(None)
                    continue
                emb = self.encoder.encode(
                    video_file=video_file,
                    start=s,
                    end=e,
                    preloaded_frames=frames,
                )
                embed_results.append(emb)

            video_file_name = os.path.basename(video_file)
            video_file_path = os.path.dirname(video_file)

            for s, e, emb, anno in zip(seg_starts, seg_ends, embed_results, annotate_results):
                if emb is None or anno is None:
                    self.stats["skipped_segments"] += 1
                    continue

                clip_id = base_metadata.get("clip_id", video_file_name)
                segment_id = f"{clip_id}_{int(s * 1000)}_{int(e * 1000)}"

                segment_metadata = {
                    **base_metadata,
                    "segment_id": segment_id,
                    "embedding_model": settings.embedding_model_version,
                    "annotation_model": settings.annotation_model_version,
                    "processing_ts": datetime.now(timezone.utc).isoformat(),
                    "clip_file_name": video_file_name,
                    "clip_file_path": video_file_path,
                }

                sql_row = {
                    "video_file_name": video_file_name,
                    "video_file_path": video_file_path,
                    "segment_start": s,
                    "segment_end": e,
                    "annotation": anno,
                    "metadata": segment_metadata,
                }
                milvus_row = {
                    "video_file_name": video_file_name,
                    "video_file_path": video_file_path,
                    "segment_start": s,
                    "segment_end": e,
                    "dataset_name": self.dataset_name,
                    "camera_channel": base_metadata.get("camera_channel", "unknown"),
                    "clip_id": clip_id,
                    "scene_token": base_metadata.get("scene_token"),
                    "embedding_model": settings.embedding_model_version,
                    "clip_vector": emb,
                }
                sql_batch.append(sql_row)
                milvus_batch.append(milvus_row)
                self.stats["inserted_segments"] += 1

        except Exception as e:
            logger.exception("Failed processing video %s", video_file)
            self.stats["failed_videos"] += 1
            return [], []

        return sql_batch, milvus_batch

    def run(self):
        video_entries = load_video_entries()
        if self.limit is not None:
            video_entries = video_entries[:self.limit]
        self.stats["total_videos"] = len(video_entries)

        sql_acc: list[dict] = []
        milvus_acc: list[dict] = []

        with tqdm(total=len(video_entries), desc="Ingestion") as pbar:
            for i in range(0, len(video_entries), settings.batch_video):
                batch = video_entries[i:i + settings.batch_video]
                parallel_out = Parallel(
                    n_jobs=settings.num_processes,
                    backend="threading",
                )(
                    delayed(self._process_single_video)(entry)
                    for entry in batch
                )

                for sq_batch, mv_batch in parallel_out:
                    sql_acc.extend(sq_batch)
                    milvus_acc.extend(mv_batch)
                    self.stats["processed_videos"] += 1

                    # flush when reach DB batch size
                    while len(sql_acc) >= settings.batch_size_db or len(milvus_acc) >= settings.batch_size_db:
                        take_sq = sql_acc[:settings.batch_size_db]
                        take_mv = milvus_acc[:settings.batch_size_db]
                        self._flush_batch(take_sq, take_mv)
                        sql_acc = sql_acc[settings.batch_size_db:]
                        milvus_acc = milvus_acc[settings.batch_size_db:]

                # gpu memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
                pbar.update(len(batch))

        # final flush
        self._flush_batch(sql_acc, milvus_acc)

        logger.info("Ingestion finished, stats: %s", self.stats)
        return self.stats
