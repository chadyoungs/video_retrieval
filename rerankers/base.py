from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class RetrievalItem:
    clip_id: str
    video_file_path: str
    coarse_score: float
    annotation: dict[str, Any]
    metadata: dict[str, Any]
    keyframe_paths: List[str]


@dataclass
class RerankedItem:
    clip_id: str
    video_file_path: str
    coarse_score: float
    rerank_score: float
    annotation: dict[str, Any]
    metadata: dict[str, Any]
    keyframe_paths: List[str]
    rerank_reason: str
    rank: int


class BaseReranker(abc.ABC):
    """Abstract base class for all rerankers."""

    def __init__(
        self,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalItem],
        annotation_conditions: Optional[dict[str, str]] = None,
        metadata_filters: Optional[dict[str, str]] = None,
        query_image_path: Optional[str] = None,
    ) -> List[RerankedItem]:
        """
        Rerank candidate retrieval items.
        :param query: text query
        :param candidates: coarse retrieval candidates
        :param annotation_conditions: annotation rule conditions (OR match)
        :param metadata_filters: metadata filter (AND match, reject on mismatch)
        :param query_image_path: optional image query path for VLM reranker
        :return: sorted RerankedItem list
        """
        ...

    @staticmethod
    def _deduplicate_candidates(candidates: List[RetrievalItem]) -> List[RetrievalItem]:
        """Deduplicate by clip_id, keep max coarse_score one."""
        clip_map: dict[str, RetrievalItem] = {}
        for item in candidates:
            if item.clip_id not in clip_map or item.coarse_score > clip_map[item.clip_id].coarse_score:
                clip_map[item.clip_id] = item
        return list(clip_map.values())

    @staticmethod
    def _build_reranked_list(
        scored_items: List[tuple[RetrievalItem, float, str]],
        top_k: int,
        threshold: float,
    ) -> List[RerankedItem]:
        """
        Shared sort + threshold + stable tie-break (clip_id) + rank assign.
        Sort: desc rerank_score, asc clip_id for tie.
        """
        filtered = [
            (item, score, reason)
            for item, score, reason in scored_items
            if score >= threshold
        ]
        filtered.sort(key=lambda x: (-x[1], x[0].clip_id))
        take = filtered[:top_k]

        out: List[RerankedItem] = []
        for idx, (item, score, reason) in enumerate(take):
            out.append(RerankedItem(
                clip_id=item.clip_id,
                video_file_path=item.video_file_path,
                coarse_score=item.coarse_score,
                rerank_score=score,
                annotation=item.annotation,
                metadata=item.metadata,
                keyframe_paths=item.keyframe_paths,
                rerank_reason=reason,
                rank=idx + 1,
            ))
        return out
