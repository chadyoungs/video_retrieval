"""Hybrid retrieval service: vector recall + metadata ranking + reranking.

Query paths
-----------
* text  -> CLIP/SigLIP text branch;
* image -> CLIP/SigLIP image branch;
* video -> pooled keyframe embedding of the query clip, same path as ingestion.

Recall
------
Milvus ANN gives dense candidates; PostgreSQL contributes candidates that
satisfy the requested metadata / annotation conditions and scores them. The two
sides are fused per *segment* and the fusion is followed by:
1. annotation back-fill for vector-only candidates (so reranking always sees
   annotation text);
2. a hard metadata filter that covers fields Milvus does not store.

The service instance is bound to one dataset, so its table and collection stay
dataset-isolated.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import settings
from db.relational_base import RelationalStoreBase
from db.store_factory import get_relational_store, get_vector_store
from db.vector_base import VectorStoreBase
from encoders.base import VideoEncoderBase
from encoders.factory import get_default_encoder, get_text_encoder
from rerankers.base import BaseReranker, RetrievalItem
from rerankers.factory import get_reranker

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch optional here
    from encoders.text_encoder import TextEncoder

logger = logging.getLogger("retrieval.service")

#: Columns carried over from a PostgreSQL row that are not "metadata" payload.
_SQL_BOOKKEEPING = {"score", "id", "created_at", "processing_ts", "annotation_model"}
#: Identity / timing fields already promoted to the record itself.
_IDENTITY_FIELDS = {
    "segment_id",
    "clip_id",
    "video_file_name",
    "video_file_path",
    "segment_start",
    "segment_end",
}


class RetrievalService:
    """Coordinate encoders, the two stores and the reranker."""

    def __init__(
        self,
        dataset_key: Optional[str] = None,
        relational_store: Optional[RelationalStoreBase] = None,
        vector_store: Optional[VectorStoreBase] = None,
        reranker: Optional[BaseReranker] = None,
        encoder: Optional[VideoEncoderBase] = None,
        text_encoder: Optional["TextEncoder"] = None,
        weight_vector: Optional[float] = None,
        weight_metadata: Optional[float] = None,
        pool_multiplier: Optional[int] = None,
    ) -> None:
        self.dataset_key = dataset_key or settings.dataset_source
        if self.dataset_key not in ("activitynet", "nuscenes"):
            raise ValueError(f"Unknown dataset_key: {self.dataset_key!r}")

        self.relational_store = relational_store or get_relational_store(self.dataset_key)
        self.vector_store = vector_store or get_vector_store(self.dataset_key)
        self.reranker = reranker or get_reranker(self.dataset_key)
        self.encoder = encoder or get_default_encoder()
        self.text_encoder = text_encoder or get_text_encoder()

        self.weight_vector = settings.weight_vector if weight_vector is None else weight_vector
        self.weight_metadata = (
            settings.weight_metadata if weight_metadata is None else weight_metadata
        )
        self.pool_multiplier = pool_multiplier or settings.recall_pool_multiplier

        logger.info(
            "Retrieval service ready for dataset=%s (dim=%s, weights=%.2f/%.2f)",
            self.dataset_key,
            getattr(self.encoder, "dim", "?"),
            self.weight_vector,
            self.weight_metadata,
        )

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
        if not query_text or not query_text.strip():
            logger.warning("Empty text query; returning empty results")
            return []

        embedding = self.text_encoder.encode(query_text)
        return self.hybrid_search(
            query_embedding=embedding,
            query_text=query_text,
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
        if not os.path.exists(query_image_path):
            logger.warning("Query image not found: %s", query_image_path)
            return []

        try:
            embedding = self.encoder.encode_image(query_image_path)
        except NotImplementedError:
            logger.error("%s cannot encode images", type(self.encoder).__name__)
            return []

        return self.hybrid_search(
            query_embedding=embedding,
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
        """Embed the query clip with the ingestion path; a middle frame feeds the VLM."""
        if not os.path.exists(query_video_path):
            logger.warning("Query video not found: %s", query_video_path)
            return []

        embedding, frames = self._encode_video_query(query_video_path, start_sec, end_sec)
        if embedding is None:
            logger.warning("Query video embedding unavailable: %s", query_video_path)
            return []

        query_image_path: Optional[str] = None
        if frames:
            query_image_path = self._save_frame(frames[len(frames) // 2])

        try:
            return self.hybrid_search(
                query_embedding=embedding,
                annotation_conditions=annotation_conditions,
                metadata_filters=metadata_filters,
                limit=limit,
                rerank=rerank,
                query_image_path=query_image_path,
            )
        finally:
            if query_image_path and os.path.exists(query_image_path):
                try:
                    os.unlink(query_image_path)
                except OSError:
                    logger.debug("Could not remove temp frame %s", query_image_path)

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
        if query_embedding is None:
            return []
        if not self._check_dim(query_embedding):
            return []

        pool_size = max(limit, settings.reranker_top_k) * self.pool_multiplier

        vec_hits = self.vector_store.search_vectors(
            query_embedding,
            top_k=pool_size,
            metadata_filters=metadata_filters,
        )
        sql_hits = self.relational_store.search_metadata(
            annotation_conditions=annotation_conditions,
            metadata_filters=metadata_filters,
            limit=pool_size,
        )

        merged = self._merge(vec_hits, sql_hits)
        self._backfill_annotations(merged)

        if metadata_filters:
            merged = {
                key: record
                for key, record in merged.items()
                if self._passes_filters(record, metadata_filters)
            }

        ordered = sorted(
            merged.values(),
            key=lambda record: (-record["final_score"], record["segment_id"]),
        )[:pool_size]

        if not rerank:
            logger.info("Hybrid recall returned %d segments (no rerank)", len(ordered))
            return self._to_output(ordered, reranked_items=None, limit=limit)

        candidates = [self._to_retrieval_item(record) for record in ordered]
        reranked = self.reranker.rerank(
            query=query_text or "",
            candidates=candidates,
            annotation_conditions=annotation_conditions,
            metadata_filters=metadata_filters,
            query_image_path=query_image_path,
        )
        logger.info("Rerank returned %d of %d candidates", len(reranked), len(candidates))
        return self._to_output(ordered, reranked)

    def _check_dim(self, embedding: Any) -> bool:
        expected = getattr(self.vector_store, "dim", None)
        actual = int(np.asarray(embedding).reshape(-1).shape[0])
        if expected and actual != expected:
            logger.error(
                "Query embedding dim %d does not match the index dim %d "
                "(model tier=%s). Re-ingest or fix semantic_model_tier.",
                actual,
                expected,
                settings.semantic_model_tier,
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    # Fusion helpers
    # ------------------------------------------------------------------ #
    def _merge(
        self,
        vec_hits: List[Dict[str, Any]],
        sql_hits: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Fuse both recall sides on ``segment_id``.

        A candidate missing from one side keeps that side's score at 0. Both
        scores are already in [0, 1], so the linear fusion stays calibrated.
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

        ann_fields = set(getattr(self.relational_store, "annotation_fields", ()))

        for hit in sql_hits:
            key = self._segment_key(hit)
            if not key:
                continue

            annotation = {
                field: hit.get(field) for field in ann_fields if hit.get(field) is not None
            }
            metadata = {
                key_name: value
                for key_name, value in hit.items()
                if key_name not in ann_fields and key_name not in _SQL_BOOKKEEPING
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

    def _backfill_annotations(self, merged: Dict[str, Dict[str, Any]]) -> None:
        """Add annotation columns for candidates that only came from Milvus."""
        missing = [
            record["segment_id"]
            for record in merged.values()
            if record["segment_id"] and not record["annotation"]
        ]
        if not missing:
            return

        try:
            rows = self.relational_store.get_segments_by_ids(missing)
        except Exception:  # noqa: BLE001 - recall must not fail on enrichment
            logger.exception("Annotation back-fill failed for %d segments", len(missing))
            return

        ann_fields = set(getattr(self.relational_store, "annotation_fields", ()))
        by_id = {row.get("segment_id"): row for row in rows}
        filled = 0
        for segment_id in missing:
            row = by_id.get(segment_id)
            if not row:
                logger.debug("Segment %s not found in the relational store", segment_id)
                continue
            record = merged.get(self._segment_key(row))
            if record is None:
                continue
            record["annotation"] = {
                field: row.get(field) for field in ann_fields if row.get(field) is not None
            }
            for key_name, value in row.items():
                if (
                    key_name in ann_fields
                    or key_name in _SQL_BOOKKEEPING
                    or key_name in _IDENTITY_FIELDS
                ):
                    continue
                record["metadata"].setdefault(key_name, value)
            filled += 1

        logger.debug("Back-filled annotations for %d/%d segments", filled, len(missing))

    @classmethod
    def _passes_filters(cls, record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, expected in filters.items():
            actual = record.get(key, record["metadata"].get(key, record["annotation"].get(key)))
            if actual is None:
                # Not stored for this dataset or segment: cannot evaluate.
                continue
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                if float(actual) != float(expected):
                    return False
            elif str(actual).strip().lower() != str(expected).strip().lower():
                return False
        return True

    @staticmethod
    def _segment_key(hit: Dict[str, Any]) -> str:
        """Segment identity: ``segment_id`` when present, else a composite key."""
        segment_id = hit.get("segment_id")
        if segment_id:
            return str(segment_id)
        clip_id = hit.get("clip_id")
        start = hit.get("segment_start")
        end = hit.get("segment_end")
        if clip_id is None and start is None:
            return ""
        return f"{clip_id}|{start}|{end}"

    @staticmethod
    def _base_record(hit: Dict[str, Any]) -> Dict[str, Any]:
        clip_id = hit.get("clip_id") or hit.get("video_file_name")
        return {
            "segment_id": str(hit.get("segment_id") or ""),
            "clip_id": str(clip_id or ""),
            "video_file_name": hit.get("video_file_name"),
            "video_file_path": hit.get("video_file_path"),
            "segment_start": hit.get("segment_start"),
            "segment_end": hit.get("segment_end"),
            "dataset_name": hit.get("dataset_name"),
            "camera_channel": hit.get("camera_channel"),
        }

    @staticmethod
    def _vec_metadata(hit: Dict[str, Any]) -> Dict[str, Any]:
        skip = {"score", "distance", "clip_vector"} | _IDENTITY_FIELDS
        return {key: value for key, value in hit.items() if key not in skip}

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
        start = start_sec or 0.0
        end = end_sec if end_sec is not None else (duration or start + settings.clip_duration_sec)
        if end <= start:
            logger.warning("Invalid query video bounds [%s, %s]", start, end)
            return None, None

        from encoders.video_loader import read_video_frames_raw

        frames = read_video_frames_raw(
            video_path, start, end, settings.frame_sampling_rate
        )
        if not frames:
            return None, None

        embedding = self.encoder.encode(
            video_path=video_path,
            start=start,
            end=end,
            preloaded_frames=frames,
        )
        return embedding, frames

    @staticmethod
    def _video_duration(video_path: str) -> float:
        import cv2

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
        import cv2

        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.close()
        cv2.imwrite(handle.name, frame)
        return handle.name

    # ------------------------------------------------------------------ #
    # Output assembly
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_retrieval_item(record: Dict[str, Any]) -> RetrievalItem:
        return RetrievalItem(
            segment_id=record["segment_id"],
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
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if reranked_items is None:
            output: List[Dict[str, Any]] = []
            for idx, record in enumerate(ordered[: limit or len(ordered)], start=1):
                output.append(
                    {
                        "rank": idx,
                        **self._record_payload(record),
                        "vec_score": record.get("vec_score", 0.0),
                        "sql_score": record.get("sql_score", 0.0),
                        "coarse_score": record.get("final_score", 0.0),
                        "final_score": record.get("final_score", 0.0),
                        "rerank_score": record.get("final_score", 0.0),
                        "rerank_reason": "coarse hybrid order",
                    }
                )
            return output

        # Reranked items are keyed by segment_id: one clip has many segments and
        # a clip_id-keyed map would leak the wrong scores into every rank.
        by_segment = {record["segment_id"]: record for record in ordered}
        output = []
        for item in reranked_items:
            record = by_segment.get(item.segment_id, {})
            payload = self._record_payload(record)
            payload.update(
                {
                    "segment_id": item.segment_id,
                    "clip_id": item.clip_id,
                    "video_file_name": item.video_file_name or payload["video_file_name"],
                    "video_file_path": item.video_file_path or payload["video_file_path"],
                    "video_full_path": self._full_path(
                        item.video_file_path, item.video_file_name
                    ),
                    "segment_start": item.segment_start,
                    "segment_end": item.segment_end,
                    "annotation": item.annotation,
                    "metadata": item.metadata,
                }
            )
            payload.update(
                {
                    "rank": item.rank,
                    "vec_score": record.get("vec_score", 0.0),
                    "sql_score": record.get("sql_score", 0.0),
                    "coarse_score": item.coarse_score,
                    "final_score": record.get("final_score", item.coarse_score),
                    "rerank_score": item.rerank_score,
                    "rerank_reason": item.rerank_reason,
                    "score_rule": item.score_rule,
                    "score_cross_encoder": item.score_cross_encoder,
                    "score_vlm": item.score_vlm,
                }
            )
            output.append(payload)
        return output

    def _record_payload(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "segment_id": record.get("segment_id"),
            "clip_id": record.get("clip_id"),
            "video_file_name": record.get("video_file_name"),
            "video_file_path": record.get("video_file_path"),
            "video_full_path": self._full_path(
                record.get("video_file_path"), record.get("video_file_name")
            ),
            "segment_start": record.get("segment_start"),
            "segment_end": record.get("segment_end"),
            "dataset_name": record.get("dataset_name"),
            "camera_channel": record.get("camera_channel"),
            "annotation": record.get("annotation") or {},
            "metadata": record.get("metadata") or {},
        }

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
