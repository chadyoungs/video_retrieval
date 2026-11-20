from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import settings
from rerankers.base import BaseReranker, RetrievalItem, RerankedItem
from config.settings import RERANK_DATASET_CONF

logger = logging.getLogger(__name__)


class IntegratedReranker(BaseReranker):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.conf = RERANK_DATASET_CONF[dataset_name]
        self.annotation_fields = self.conf["annotation_fields"]
        self.rule_match_mode = self.conf["rule_match_mode"]

        self.threshold = settings.reranker_score_threshold
        self.top_k = settings.reranker_top_k

        self.w_rule = settings.reranker_weight_rule
        self.w_ce = settings.reranker_weight_cross_encoder
        self.w_vlm = settings.reranker_weight_vlm

        self.cross_encoder_name = settings.cross_encoder_model_name
        self.ce_max_seq_len = 512
        self.ce_batch_size = 16
        self.ce_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ce_tokenizer: Optional[AutoTokenizer] = None
        self._ce_model: Optional[AutoModelForSequenceClassification] = None

        self.ollama_host = settings.ollama_host
        self.ollama_rerank_model = settings.ollama_rerank_model

    def _load_cross_encoder(self):
        if self._ce_model is not None:
            return
        logger.info(f"Loading cross encoder: {self.cross_encoder_name} to {self.ce_device}")
        self._ce_tokenizer = AutoTokenizer.from_pretrained(self.cross_encoder_name)
        self._ce_model = AutoModelForSequenceClassification.from_pretrained(
            self.cross_encoder_name
        ).to(self.ce_device)
        self._ce_model.eval()

    def _rule_gate(self, item: RetrievalItem, filter_spec: Dict[str, Any]) -> bool:
        """Rule 闸门：元数据条件 AND，任一不满足直接丢弃"""
        filter_meta = {k: v for k, v in filter_spec.items() if k not in self.annotation_fields}
        for k, expect_val in filter_meta.items():
            val = item.metadata.get(k)
            if val is None:
                return False
            if str(val).lower() != str(expect_val).lower():
                return False
        return True

    def _score_rule(self, item: RetrievalItem, filter_spec: Dict[str, Any]) -> float:
        """规则软打分：标注字段 OR 命中比例"""
        filter_annot = {k: v for k, v in filter_spec.items() if k in self.annotation_fields}
        if not filter_annot:
            return 1.0
        annot_subset = {k: item.annotation.get(k) for k in self.annotation_fields}
        hit = 0
        total = len(filter_annot)
        for k, expect_val in filter_annot.items():
            val = annot_subset.get(k)
            if val is None:
                continue
            if str(val).lower() == str(expect_val).lower():
                hit += 1
        return hit / total

    def _score_cross_encoder_batch(self, query: str, items: List[RetrievalItem]) -> List[float]:
        """批量 cross encoder 打分，NaN/Inf 兜底，文本拼接自 annotation"""
        self._load_cross_encoder()
        pairs = []
        index_map = []
        for idx, it in enumerate(items):
            doc_text = "; ".join(f"{k}:{v}" for k, v in it.annotation.items() if v is not None)
            doc_text = doc_text.strip()
            if not doc_text:
                logger.warning(f"Clip {it.clip_id}: empty annotation text for CE, fallback coarse")
                continue
            pairs.append((query, doc_text))
            index_map.append(idx)

        scores = [it.coarse_score for it in items]
        if not pairs:
            return scores

        all_ce_scores = []
        try:
            with torch.no_grad():
                for start in range(0, len(pairs), self.ce_batch_size):
                    batch = pairs[start:start + self.ce_batch_size]
                    inputs = self._ce_tokenizer(
                        batch,
                        truncation=True,
                        max_length=self.ce_max_seq_len,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.ce_device)
                    out = self._ce_model(**inputs)
                    logits = out.logits[:, 0]
                    batch_scores = torch.sigmoid(logits).cpu().numpy()
                    batch_scores = np.nan_to_num(batch_scores, nan=0.0, posinf=1.0, neginf=0.0)
                    all_ce_scores.extend(batch_scores.tolist())
        except Exception as e:
            logger.error(f"Cross encoder batch inference failed: {e}, fallback coarse", exc_info=True)
            return scores

        for pos, orig_idx in enumerate(index_map):
            scores[orig_idx] = all_ce_scores[pos]
        return scores

    def _score_vlm_single(self, query: str, item: RetrievalItem) -> float:
        """VLM 单条打分，0~10 -> 0~1，异常兜底 coarse_score"""
        keyframe_paths = item.metadata.get("keyframe_paths", [])
        if not keyframe_paths:
            logger.warning(f"Clip {item.clip_id}: no keyframe for VLM rerank, fallback coarse")
            return item.coarse_score

        prompt = (
            f"判断视频片段与查询的相关性，只输出0~10的整数分数，不要解释、不要额外文字。\n"
            f"Query: {query}\n"
            f"Annotation: {item.annotation}"
        )
        payload = {
            "model": self.ollama_rerank_model,
            "prompt": prompt,
            "images": keyframe_paths[:1],
            "stream": False
        }
        try:
            resp = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=20)
            resp.raise_for_status()
            raw = resp.json()["response"].strip()
            score_val = float(raw)
            norm = max(0.0, min(10.0, score_val)) / 10.0
            return norm
        except Exception as e:
            logger.warning(f"VLM rerank failed for clip {item.clip_id}: {e}, fallback coarse")
            return item.coarse_score

    def _wrap_coarse(self, candidates: List[RetrievalItem]) -> List[RerankedItem]:
        """兜底透传：直接使用粗排分作为 rerank_score"""
        sorted_candidates = sorted(candidates, key=lambda x: (-x.coarse_score, x.clip_id))
        out = []
        for rank_idx, it in enumerate(sorted_candidates[:self.top_k]):
            out.append(RerankedItem(
                **it.model_dump(),
                rerank_score=it.coarse_score,
                rank=rank_idx + 1
            ))
        return out

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalItem],
        filter_spec: Dict[str, Any]
    ) -> List[RerankedItem]:
        # 1. 空候选
        if not candidates:
            logger.debug("Rerank input candidates empty")
            return []

        # 2. clip_id 去重，保留粗排最高
        dedup_map: Dict[str, RetrievalItem] = {}
        for it in candidates:
            cid = it.clip_id
            if cid not in dedup_map or it.coarse_score > dedup_map[cid].coarse_score:
                dedup_map[cid] = it
        unique_candidates = list(dedup_map.values())
        if len(unique_candidates) < len(candidates):
            logger.warning(f"Duplicate clip_id, dedup {len(candidates)} -> {len(unique_candidates)}")

        # 3. 空查询，直接透传粗排
        query = query.strip()
        if not query:
            logger.warning("Empty query, skip rerank and fallback coarse")
            return self._wrap_coarse(unique_candidates)

        # 4. Rule 闸门过滤（元数据 AND）
        after_gate = [it for it in unique_candidates if self._rule_gate(it, filter_spec)]
        if not after_gate:
            logger.info("All candidates rejected by rule gate")
            return []

        # 5. Rule 软打分
        rule_scores = [self._score_rule(it, filter_spec) for it in after_gate]
        # 6. Cross-encoder 批量打分
        ce_scores = self._score_cross_encoder_batch(query, after_gate)
        # 7. VLM 逐个打分
        vlm_scores = [self._score_vlm_single(query, it) for it in after_gate]

        # 8. 加权融合
        scored_items = []
        for it, s_rule, s_ce, s_vlm in zip(after_gate, rule_scores, ce_scores, vlm_scores):
            fused = s_rule * self.w_rule + s_ce * self.w_ce + s_vlm * self.w_vlm
            scored_items.append((it, fused, s_rule, s_ce, s_vlm))

        # 9. 阈值过滤
        filtered = [entry for entry in scored_items if entry[1] >= self.threshold]
        if not filtered:
            logger.info(f"No candidates above rerank threshold {self.threshold}")
            return []

        # 10. 排序：融合分降序，同分按 clip_id 稳定
        filtered.sort(key=lambda x: (-x[1], x[0].clip_id))

        # 11. Top-K 截断
        selected = filtered[:self.top_k]

        # 12. 组装输出 RerankedItem
        out = []
        for rank_idx, (it, fused, s_rule, s_ce, s_vlm) in enumerate(selected):
            payload = it.model_dump()
            payload["score_rule"] = s_rule
            payload["score_cross_encoder"] = s_ce
            payload["score_vlm"] = s_vlm
            out.append(RerankedItem(
                **payload,
                rerank_score=fused,
                rank=rank_idx + 1
            ))
        return out
