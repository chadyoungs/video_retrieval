"""Hybrid retrieval service: vector recall + metadata ranking + reranking.

The service digests the legacy ``retrieval`` module:

* text query    -> CLIP text branch embedding;
* image query   -> CLIP image branch embedding;
* video query   -> pooled keyframe / sampled-frame embedding of the query clip;
* hybrid recall -> Milvus ANN candidates fused with PostgreSQL metadata /
  annotation ranking on a shared segment key;
* reranking     -> integrated rule gate + cross-encoder + VLM fused reranker.

One service instance is bound to one dataset, so its relational store and
vector collection stay dataset-isolated.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2

from config.settings import settings
from db.relational_base import RelationalStoreBase
from db.store_factory import get_relational_store, get_vector_store
from db.vector_base import VectorStoreBase
from encoders.base import VideoEncoderBase
from encoders.factory import get_encoder, get_text_encoder
from encoders.text_encoder import TextEncoder
from encoders.video_loader import read_video_frames_raw
from rerankers.base import BaseReranker, RetrievalItem
from rerankers.factory import get_reranker

logger = logging.getLogger("app.retrieval")


class RetrievalService:
    """Coordinate encoders, the two stores and the reranker."""

    def __init__(
        self,
        dataset_key: Optional[str] = None,
        relational_store: Optional[RelationalStoreBase] = None,
        vector_store: Optional[VectorStoreBase] = None,
        reranker: Optional[BaseReranker] = None,
        encoder: Optional[VideoEncoderBase] = None,
        text_encoder: Optional[TextEncoder] = None,
        weight_vector: float = 0.7,
        weight_metadata: float = 0.3,
        pool_multiplier: int = 2,
    ) -> None:
        self.dataset_key = dataset_key or settings.dataset_source
        self.relational_store = (
            relational_store or get_relational_store(self.dataset_key)
        )
        self.vector_store = vector_store or get_vector_store(self.dataset_key)
        self.reranker = reranker or get_reranker(self.dataset_key)
        self.encoder = encoder or get_encoder()
        self.text_encoder = text_encoder or get_text_encoder()

        self.weight_vector = weight_vector
        self.weight_metadata = weight_metadata
        # Recall pool = max(limit, reranker_candidates) * multiplier. A wider
        # pool before reranking improves rerank precision at modest recall cost.
        self.pool_multiplier = pool_multiplier

    # ------------------------------------------------------------------ #
    # Public query entries
    # ------------------------------------------------------------------ #
    def search_text(
        self,
        query_text: str,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve segments matching a natural-language query."""
        if not query_text or not query_text.strip():
            logger.warning("Empty text query; returning empty results")
            return []

        embedding = self.text_encoder.encode(query_text)
        return self.hybrid_search(
            query_embedding=embedding,
            query_text=query_text,
            query_image_path=None,
            annotation_conditions=annotation_conditions,
            metadata_filters=metadata_filters,
            limit=limit,
            rerank=rerank,
        )

    def search_image(
        self,
        query_image_path: str,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve segments visually matching a reference image."""
        if not os.path.exists(query_image_path):
            logger.warning("Query image not found: %s", query_image_path)
            return []

        embedding = self.encoder.encode_image(query_image_path)
        return self.hybrid_search(
            query_embedding=embedding,
            query_text="",
            query_image_path=query_image_path,
            annotation_conditions=annotation_conditions,
            metadata_filters=metadata_filters,
            limit=limit,
            rerank=rerank,
        )

    def search_video(
        self,
        query_video_path: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve segments matching a query video clip.

        The whole clip (or ``[start_sec, end_sec)``) is embedded with the same
        frame-pooling path used at ingestion. A middle frame is exported for
        the VLM reranker's reference-image branch.
        """
        if not os.path.exists(query_video_path):
            logger.warning("Query video not found: %s", query_video_path)
            return []

        embedding, frames = self._encode_video_query(
            query_video_path, start_sec, end_sec
        )
        if embedding is None:
            logger.warning(
                "Query video embedding unavailable: %s", query_video_path
            )
            return []

        query_image_path: Optional[str] = None
        if frames:
            middle = frames[len(frames) // 2]
            query_image_path = self._save_frame(middle)

        return self.hybrid_search(
            query_embedding=embedding,
            query_text="",
            query_image_path=query_image_path,
            annotation_conditions=annotation_conditions,
            metadata_filters=metadata_filters,
            limit=limit,
            rerank=rerank,
        )

    # ------------------------------------------------------------------ #
    # Hybrid recall + fusion
    # ------------------------------------------------------------------ #
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str = "",
        query_image_path: Optional[str] = None,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run vector + metadata recall, fuse, then optionally rerank."""
        if query_embedding is None:
            return []

        pool_size = max(limit, settings.reranker_candidates) * self.pool_multiplier

        vec_hits = self.vector_store.search_vectors(
            query_embedding,
            limit=pool_size,
            metadata_filters=metadata_filters,
        )
        sql_hits = self.relational_store.search_metadata(
            annotation_conditions=annotation_conditions,
            metadata_filters=metadata_filters,
            limit=pool_size,
        )

        merged = self._merge(vec_hits, sql_hits)
        ordered = sorted(
            merged.values(),
            key=lambda record: (-record["final_score"], record["clip_id"]),
        )[:pool_size]

        if not rerank:
            return self._to_output(ordered, reranked_items=None)

        candidates = [self._to_retrieval_item(record) for record in ordered]
        reranked = self.reranker.rerank(
            query=query_text or "",
            candidates=candidates,
            annotation_conditions=annotation_conditions,
            metadata_filters=metadata_filters,
            query_image_path=query_image_path,
        )
        return self._to_output(ordered, reranked)

    # ------------------------------------------------------------------ #
    # Fusion helpers
    # ------------------------------------------------------------------ #
    def _merge(
        self,
        vec_hits: List[Dict[str, Any]],
        sql_hits: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Fuse vector and SQL hits on the shared segment key.

        A candidate missing from one side keeps that side's score at 0; it is
        not treated as a perfect match. Scores on both sides are already in
        [0, 1], so the linear weighted fusion stays calibrated.
        """
        merged: Dict[str, Dict[str, Any]] = {}

        for hit in vec_hits:
            key = self._segment_key(hit)
            if not key:
                continue
            record = self._base_record(hit)
            record["vec_score"] = float(hit.get("score") or 0.0)
            record["sql_score"] = 0.0
            record["distance"] = hit.get("distance")
            record["annotation"] = {}
            record["metadata"] = self._vec_metadata(hit)
            record["final_score"] = self.weight_vector * record["vec_score"]
            merged[key] = record

        ann_fields = set(
            getattr(self.relational_store, "annotation_fields", ())
        )
        skip_sql_columns = {"score", "id", "created_at"}

        for hit in sql_hits:
            key = self._segment_key(hit)
            if not key:
                continue

            annotation = {
                field: hit.get(field)
                for field in ann_fields
                if hit.get(field) is not None
            }
            metadata = {
                k: v
                for k, v in hit.items()
                if k not in ann_fields and k not in skip_sql_columns
            }
            sql_score = float(hit.get("score") or 0.0)

            if key in merged:
                record = merged[key]
                record["sql_score"] = sql_score
                record["annotation"].update(annotation)
                record["metadata"].update(metadata)
                record["final_score"] = (
                    self.weight_vector * record["vec_score"]
                    + self.weight_metadata * sql_score
                )
            else:
                record = self._base_record(hit)
                record["vec_score"] = 0.0
                record["sql_score"] = sql_score
                record["annotation"] = annotation
                record["metadata"] = metadata
                record["final_score"] = self.weight_metadata * sql_score
                merged[key] = record

        return merged

    @staticmethod
    def _segment_key(hit: Dict[str, Any]) -> str:
        clip_id = hit.get("clip_id")
        start = hit.get("segment_start")
        end = hit.get("segment_end")
        if clip_id is None and start is None:
            return ""
        return f"{clip_id}|{start}|{end}"

    @staticmethod
    def _base_record(hit: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "clip_id": str(hit.get("clip_id") or hit.get("video_file_name") or ""),
            "video_file_name": hit.get("video_file_name"),
            "video_file_path": hit.get("video_file_path"),
            "segment_start": hit.get("segment_start"),
            "segment_end": hit.get("segment_end"),
            "dataset_name": hit.get("dataset_name"),
            "camera_channel": hit.get("camera_channel"),
        }

    @staticmethod
    def _vec_metadata(hit: Dict[str, Any]) -> Dict[str, Any]:
        skip = {
            "score",
            "distance",
            "clip_vector",
            "video_file_name",
            "video_file_path",
            "segment_start",
            "segment_end",
            "clip_id",
        }
        return {k: v for k, v in hit.items() if k not in skip}

    # ------------------------------------------------------------------ #
    # Query video encoding
    # ------------------------------------------------------------------ #
    def _encode_video_query(
        self,
        video_path: str,
        start_sec: Optional[float],
        end_sec: Optional[float],
    ) -> Tuple[Optional[List[float]], Optional[List[Any]]]:
        duration = self._video_duration(video_path)
        if duration <= 0:
            start = start_sec or 0.0
            end = end_sec or 0.0
        else:
            start = start_sec or 0.0
            end = end_sec if end_sec is not None else duration

        frames = read_video_frames_raw(
            video_path, start, end, settings.frame_sampling_rate
        )
        if not frames:
            return None, None

        embedding = self.encoder.encode(
            video_path=video_path,
            start_sec=start,
            end_sec=end,
            preloaded_frames=frames,
        )
        return embedding, frames

    @staticmethod
    def _video_duration(video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps <= 0 or total_frames <= 0:
                return 0.0
            return float(total_frames / fps)
        finally:
            cap.release()

    @staticmethod
    def _save_frame(frame: Any) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, frame)
        return tmp.name

    # ------------------------------------------------------------------ #
    # Output assembly
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_retrieval_item(record: Dict[str, Any]) -> RetrievalItem:
        return RetrievalItem(
            clip_id=record["clip_id"],
            video_file_path=record.get("video_file_path") or "",
            video_file_name=record.get("video_file_name"),
            segment_start=record.get("segment_start"),
            segment_end=record.get("segment_end"),
            annotation=record.get("annotation") or {},
            metadata=record.get("metadata") or {},
            coarse_score=record["final_score"],
            keyframe_paths=record.get("keyframe_paths", []),
        )

    def _to_output(
        self,
        ordered: List[Dict[str, Any]],
        reranked_items: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        if reranked_items is None:
            output: List[Dict[str, Any]] = []
            for idx, record in enumerate(ordered):
                row = dict(record)
                row["rank"] = idx + 1
                row["rerank_score"] = record["final_score"]
                row["rerank_reason"] = "coarse hybrid order"
                output.append(row)
            return output

        by_clip = {record["clip_id"]: record for record in ordered}
        output = []
        for item in reranked_items:
            record = by_clip.get(item.clip_id, {})
            output.append(
                {
                    "rank": item.rank,
                    "clip_id": item.clip_id,
                    "video_file_name": item.video_file_name,
                    "video_file_path": item.video_file_path,
                    "video_full_path": self._full_path(
                        item.video_file_path, item.video_file_name
                    ),
                    "segment_start": item.segment_start,
                    "segment_end": item.segment_end,
                    "dataset_name": item.metadata.get("dataset_name"),
                    "camera_channel": item.metadata.get("camera_channel"),
                    "annotation": item.annotation,
                    "metadata": item.metadata,
                    "vec_score": record.get("vec_score", 0.0),
                    "sql_score": record.get("sql_score", 0.0),
                    "coarse_score": item.coarse_score,
                    "final_score": record.get(
                        "final_score", item.coarse_score
                    ),
                    "rerank_score": item.rerank_score,
                    "rerank_reason": item.rerank_reason,
                }
            )
        return output

    @staticmethod
    def _full_path(directory: Optional[str], name: Optional[str]) -> str:
        if directory and name:
            return os.path.join(directory, name)
        return directory or name or ""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        for store in (self.relational_store, self.vector_store):
            try:
                store.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                logger.exception("Error closing %s", type(store).__name__)
