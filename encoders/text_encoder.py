"""Standalone text-query encoder.

It loads the *same* model family as the video encoder (CLIP or SigLIP, chosen by
``settings.semantic_model_tier``) so text and clip vectors share one space and
one dimension. Loading a different model here silently breaks ANN search.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from config.settings import settings
from utils.device import resolve_device
from utils.hf_models import load_model, load_processor

logger = logging.getLogger(__name__)


class TextEncoder:
    def __init__(self) -> None:
        self.device = resolve_device()
        self.model_name = settings.clip_model_name

        logger.info("Loading text encoder %s on %s", self.model_name, self.device)
        self.model = load_model(self.model_name, AutoModel)
        self.processor = load_processor(self.model_name, AutoProcessor)
        self.model.to(self.device).eval()
        self._dim = int(
            getattr(self.model.config, "projection_dim", 0) or settings.embedding_dim
        )

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, ord=2)
        if norm < 1e-8:
            return vec
        return vec / norm

    def encode(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(
                text=text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            features = self.model.get_text_features(**inputs)
            arr = features.detach().cpu().numpy()[0]
        return self._l2_normalize(arr)
