"""Reranker contract and shared candidate dataclasses.

A candidate is one *segment*, identified by ``segment_id``. ``clip_id`` is the
source clip/video and is only used for display grouping, never as a key.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalItem:
    """One coarse-retrieval candidate handed to a reranker."""

    segment_id: str
    clip_id: str
    video_file_path: str
    coarse_score: float
    annotation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    video_file_name: Optional[str] = None
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    keyframe_paths: List[str] = field(default_factory=list)


@dataclass
class RerankedItem:
    """A reranked candidate with component-level score breakdown."""

    segment_id: str
    clip_id: str
    video_file_path: str
    coarse_score: float
    rerank_score: float
    rerank_reason: str
    rank: int
    annotation: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    video_file_name: Optional[str] = None
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    keyframe_paths: List[str] = field(default_factory=list)
    score_rule: Optional[float] = None
    score_cross_encoder: Optional[float] = None
    score_vlm: Optional[float] = None


class BaseReranker(abc.ABC):
    """Abstract reranker interface."""

    def __init__(self, top_k: int = 10, score_threshold: float = 0.0):
        self.top_k = top_k
        self.score_threshold = score_threshold

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalItem],
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        query_image_path: Optional[str] = None,
    ) -> List[RerankedItem]:
        """Re-score and re-order candidates.

        Args:
            query: text query (may be empty for image / video queries).
            candidates: coarse retrieval candidates.
            annotation_conditions: annotation rule conditions, OR semantics.
            metadata_filters: metadata conditions, AND semantics, hard gate.
            query_image_path: image query (or video middle frame) for VLM.
        """

    @staticmethod
    def deduplicate(candidates: List[RetrievalItem]) -> List[RetrievalItem]:
        """Keep the highest-coarse-score candidate per ``segment_id``."""
        kept: Dict[str, RetrievalItem] = {}
        for item in candidates:
            current = kept.get(item.segment_id)
            if current is None or item.coarse_score > current.coarse_score:
                kept[item.segment_id] = item
        return list(kept.values())

    @staticmethod
    def build_reranked_list(
        scored_items: List[tuple[RetrievalItem, float, str]],
        top_k: int,
        threshold: float,
    ) -> List[RerankedItem]:
        """Sort by score, apply the threshold, assign ranks."""
        filtered = [entry for entry in scored_items if entry[1] >= threshold]
        # Score first; on ties prefer the stronger coarse candidate, then a
        # stable segment_id tie-break so identical requests stay reproducible.
        filtered.sort(
            key=lambda entry: (-entry[1], -entry[0].coarse_score, entry[0].segment_id)
        )

        out: List[RerankedItem] = []
        for rank, (item, score, reason) in enumerate(filtered[:top_k], start=1):
            out.append(
                RerankedItem(
                    segment_id=item.segment_id,
                    clip_id=item.clip_id,
                    video_file_path=item.video_file_path,
                    coarse_score=item.coarse_score,
                    rerank_score=score,
                    rerank_reason=reason,
                    rank=rank,
                    annotation=item.annotation,
                    metadata=item.metadata,
                    video_file_name=item.video_file_name,
                    segment_start=item.segment_start,
                    segment_end=item.segment_end,
                    keyframe_paths=item.keyframe_paths,
                )
            )
        return out
