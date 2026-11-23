"""Integrated reranker: rule gate + rule score + cross-encoder + VLM.

Design notes
------------
* Every component is optional. A component that is disabled, has no local
  weights, or whose endpoint is unreachable is *skipped* and the remaining
  weights are renormalised. Padding a missing component with the coarse score
  (the legacy behaviour) silently cancels out the reranking.
* The rule gate is a hard filter on ``metadata_filters`` (AND). The rule score
  is the soft OR-match ratio over ``annotation_conditions``.
* The VLM branch scores the candidate's own frame extracted from the segment
  (keyframe paths are not persisted yet) and optionally the query image.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from annotators.rules import ANNOTATION_RULES_ACTIVITYNET, ANNOTATION_RULES_NUSCENES
from config.settings import settings
from rerankers.base import BaseReranker, RerankedItem, RetrievalItem

logger = logging.getLogger(__name__)

_ANNOTATION_FIELDS: Dict[str, Tuple[str, ...]] = {
    "activitynet": tuple(ANNOTATION_RULES_ACTIVITYNET.keys()),
    "nuscenes": tuple(ANNOTATION_RULES_NUSCENES.keys()),
}


class IntegratedReranker(BaseReranker):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(
            top_k=settings.reranker_top_k,
            score_threshold=settings.reranker_score_threshold,
        )
        if dataset_name not in _ANNOTATION_FIELDS:
            raise ValueError(f"Unknown dataset_name: {dataset_name!r}")

        self.dataset_name = dataset_name
        self.conf = settings.rerank_conf(dataset_name)
        self.annotation_fields = _ANNOTATION_FIELDS[dataset_name]

        self.w_rule = float(self.conf["rule_weight"])
        self.w_ce = float(self.conf["ce_weight"])
        self.w_vlm = float(self.conf["vlm_weight"])

        self.cross_encoder_name = settings.cross_encoder_model_name
        self.ce_max_seq_len = settings.ce_max_seq_len
        self.ce_batch_size = settings.ce_batch_size
        self.ce_device = None
        self._ce_tokenizer = None
        self._ce_model = None

        self.ollama_url = settings.ollama_generate_url
        self.ollama_model = settings.ollama_rerank_model or settings.ollama_model
        self.ollama_timeout = settings.ollama_rerank_timeout

        # Runtime availability flags; flipped off after the first hard failure.
        self._ce_available = settings.reranker_enable_cross_encoder
        self._vlm_available = settings.reranker_enable_vlm
        self._frame_b64_cache: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------ #
    # Cross encoder
    # ------------------------------------------------------------------ #
    def _load_cross_encoder(self) -> bool:
        if self._ce_model is not None:
            return True
        if not self._ce_available:
            return False
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from utils.device import resolve_device
            from utils.hf_models import load_model, load_processor

            self.ce_device = resolve_device()
            logger.info(
                "Loading cross encoder %s on %s", self.cross_encoder_name, self.ce_device
            )
            self._ce_tokenizer = load_processor(
                self.cross_encoder_name, AutoTokenizer
            )
            self._ce_model = load_model(
                self.cross_encoder_name, AutoModelForSequenceClassification
            ).to(self.ce_device)
            self._ce_model.eval()
            self._torch = torch
            return True
        except Exception as exc:  # noqa: BLE001 - optional component
            logger.warning(
                "Cross encoder unavailable (%s); rerank continues without it", exc
            )
            self._ce_available = False
            return False

    def _score_cross_encoder(self, query: str, items: List[RetrievalItem]) -> Optional[List[float]]:
        """Batched cross-encoder scores; ``None`` when the component is unusable."""
        if not query.strip() or not self._load_cross_encoder():
            return None

        pairs: List[Tuple[str, str]] = []
        index_map: List[int] = []
        for idx, item in enumerate(items):
            doc = self._candidate_text(item)
            if not doc:
                logger.debug("Segment %s has no annotation text for CE", item.segment_id)
                continue
            pairs.append((query, doc))
            index_map.append(idx)

        if not pairs:
            return None

        torch = self._torch
        scores: List[Optional[float]] = [None] * len(items)
        try:
            with torch.no_grad():
                for start in range(0, len(pairs), self.ce_batch_size):
                    batch = pairs[start : start + self.ce_batch_size]
                    inputs = self._ce_tokenizer(
                        batch,
                        truncation=True,
                        max_length=self.ce_max_seq_len,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.ce_device)
                    logits = self._ce_model(**inputs).logits[:, 0]
                    batch_scores = torch.sigmoid(logits).cpu().numpy()
                    batch_scores = np.nan_to_num(
                        batch_scores, nan=0.0, posinf=1.0, neginf=0.0
                    )
                    for offset, score in enumerate(batch_scores.tolist()):
                        scores[index_map[start + offset]] = float(score)
        except Exception:  # noqa: BLE001 - optional component
            logger.exception("Cross encoder inference failed; skipping component")
            self._ce_available = False
            return None

        return [score if score is not None else 0.0 for score in scores]

    # ------------------------------------------------------------------ #
    # VLM
    # ------------------------------------------------------------------ #
    def _score_vlm(
        self,
        query: str,
        items: List[RetrievalItem],
        query_image_path: Optional[str],
    ) -> Optional[List[float]]:
        """VLM 0-10 relevance per candidate; ``None`` when unavailable."""
        if not self._vlm_available:
            return None

        import requests

        query_image_b64 = self._image_path_to_b64(query_image_path)
        scores: List[float] = []
        scored_any = False

        for item in items:
            frame_b64 = self._candidate_frame_b64(item)
            if not frame_b64:
                scores.append(0.0)
                continue
            scored_any = True

            prompt = (
                "Rate how well this video segment matches the query on a scale of "
                "0 to 10. Output the integer only, no explanation.\n"
                f"Query: {query or 'visually similar frame'}\n"
                f"Annotation: {self._candidate_text(item)}"
            )
            images = [query_image_b64, frame_b64] if query_image_b64 else [frame_b64]
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "images": images,
                "stream": False,
            }
            try:
                resp = requests.post(
                    self.ollama_url, json=payload, timeout=self.ollama_timeout
                )
                resp.raise_for_status()
                raw = str(resp.json().get("response", "")).strip()
                value = float(raw.split()[0])
                scores.append(max(0.0, min(10.0, value)) / 10.0)
            except Exception as exc:  # noqa: BLE001 - optional component
                logger.warning(
                    "VLM rerank unavailable (%s); skipping component for this run", exc
                )
                self._vlm_available = False
                return None

        if not scored_any:
            logger.warning("No candidate frame readable; VLM rerank skipped")
            return None
        return scores

    def _candidate_frame_b64(self, item: RetrievalItem) -> Optional[str]:
        """One JPEG frame from the middle of the candidate segment, base64-encoded."""
        cache_key = item.segment_id
        if cache_key in self._frame_b64_cache:
            return self._frame_b64_cache[cache_key]

        result: Optional[str] = None
        try:
            from encoders.video_loader import read_video_frames_raw

            full_path = self._full_path(item)
            if not full_path or not os.path.exists(full_path):
                logger.warning("Candidate video missing on disk: %s", full_path)
            else:
                mid = None
                if item.segment_start is not None and item.segment_end is not None:
                    mid = (float(item.segment_start) + float(item.segment_end)) / 2.0
                frames = read_video_frames_raw(
                    video_path=full_path,
                    start=max(0.0, (mid or 0.0)),
                    end=(mid or 0.0) + 1.0,
                    sample_rate=1,
                )
                if frames:
                    buffer = io.BytesIO()
                    frames[0].convert("RGB").save(buffer, format="JPEG", quality=85)
                    result = base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:  # noqa: BLE001 - optional component
            logger.exception("Could not extract VLM frame for %s", item.segment_id)

        self._frame_b64_cache[cache_key] = result
        return result

    @staticmethod
    def _image_path_to_b64(path: Optional[str]) -> Optional[str]:
        if not path or not os.path.exists(path):
            return None
        try:
            from PIL import Image

            with Image.open(path) as image:
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:  # noqa: BLE001
            logger.exception("Could not read query image %s", path)
            return None

    # ------------------------------------------------------------------ #
    # Rules
    # ------------------------------------------------------------------ #
    @staticmethod
    def _full_path(item: RetrievalItem) -> str:
        if item.video_file_path and item.video_file_name:
            return os.path.join(item.video_file_path, item.video_file_name)
        return item.video_file_path or item.video_file_name or ""

    def _candidate_text(self, item: RetrievalItem) -> str:
        parts = [f"{key}:{value}" for key, value in item.annotation.items() if value]
        summary = item.metadata.get("semantic_summary") if item.metadata else None
        if summary and summary not in parts:
            parts.append(f"summary:{summary}")
        return "; ".join(parts)

    def _gate(self, item: RetrievalItem, metadata_filters: Dict[str, Any]) -> Optional[bool]:
        """Hard AND gate. Returns None when the key is absent from this dataset."""
        for key, expected in metadata_filters.items():
            actual = item.metadata.get(key, item.annotation.get(key))
            if actual is None:
                # Field not materialised for this dataset: cannot evaluate.
                continue
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                if float(actual) != float(expected):
                    return False
            elif str(actual).strip().lower() != str(expected).strip().lower():
                return False
        return True

    def _score_rule(self, item: RetrievalItem, annotation_conditions: Dict[str, Any]) -> Optional[float]:
        conditions = {
            key: value
            for key, value in annotation_conditions.items()
            if key in self.annotation_fields
        }
        if not conditions:
            return None

        hits = 0
        for key, expected in conditions.items():
            actual = item.annotation.get(key)
            if actual is not None and str(actual).strip().lower() == str(expected).strip().lower():
                hits += 1
        return hits / len(conditions)

    # ------------------------------------------------------------------ #
    # Entry
    # ------------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalItem],
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        query_image_path: Optional[str] = None,
    ) -> List[RerankedItem]:
        if not candidates:
            logger.debug("Rerank input candidates empty")
            return []

        unique = self.deduplicate(candidates)
        if len(unique) < len(candidates):
            logger.debug("Deduplicated %d -> %d segments", len(candidates), len(unique))

        filters = dict(metadata_filters or {})
        gated = [item for item in unique if self._gate(item, filters) is not False]
        if len(gated) < len(unique):
            logger.info(
                "Rule gate dropped %d of %d candidates", len(unique) - len(gated), len(unique)
            )
        if not gated:
            logger.info("All candidates rejected by the metadata rule gate")
            return []

        conditions = dict(annotation_conditions or {})
        guard = len(gated)

        rule_scores = [self._score_rule(item, conditions) for item in gated]
        ce_scores = self._score_cross_encoder(query.strip(), gated)
        vlm_scores = self._score_vlm(query.strip(), gated, query_image_path)

        weights = {
            "rule": (self.w_rule if rule_scores[0] is not None else 0.0),
            "ce": (self.w_ce if ce_scores is not None else 0.0),
            "vlm": (self.w_vlm if vlm_scores is not None else 0.0),
        }
        total_weight = sum(weights.values())

        if total_weight <= 0:
            logger.warning("No rerank signal available; falling back to coarse order")
            scored = [
                (item, item.coarse_score, "coarse: no rerank signal") for item in gated
            ]
            return self.build_reranked_list(scored, self.top_k, self.score_threshold)

        scored_items: List[Tuple[RetrievalItem, float, str]] = []
        for idx in range(guard):
            fused = 0.0
            reasons = []
            if weights["rule"]:
                fused += weights["rule"] * (rule_scores[idx] or 0.0)
                reasons.append(f"rule={rule_scores[idx]:.2f}")
            if weights["ce"]:
                fused += weights["ce"] * (ce_scores[idx] or 0.0)
                reasons.append(f"ce={ce_scores[idx]:.2f}")
            if weights["vlm"]:
                fused += weights["vlm"] * (vlm_scores[idx] or 0.0)
                reasons.append(f"vlm={vlm_scores[idx]:.2f}")
            fused /= total_weight
            reasons.append(f"coarse={gated[idx].coarse_score:.2f}")
            scored_items.append((gated[idx], fused, " ".join(reasons)))

        reranked = self.build_reranked_list(scored_items, self.top_k, self.score_threshold)
        # Re-attach the component breakdown, which the generic builder drops.
        by_segment = {
            item.segment_id: (rule, ce, vlm)
            for item, (rule, ce, vlm) in zip(
                gated,
                zip(
                    rule_scores or [None] * guard,
                    ce_scores or [None] * guard,
                    vlm_scores or [None] * guard,
                ),
            )
        }
        for entry in reranked:
            rule, ce, vlm = by_segment.get(entry.segment_id, (None, None, None))
            entry.score_rule = rule
            entry.score_cross_encoder = ce
            entry.score_vlm = vlm

        if not reranked:
            logger.info("No candidates passed rerank threshold %.3f", self.score_threshold)
        return reranked
