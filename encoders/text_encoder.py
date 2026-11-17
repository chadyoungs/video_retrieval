from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.config.settings import settings

logger = logging.getLogger(__name__)


class TextEncoder:
    """
    Standalone text encoder for retrieval query.
    Separate from video encoder, share same CLIP model family.
    """
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = settings.base_clip_model_name

        logger.info(f"Loading TextEncoder CLIP: {model_path} on {self.device}")
        self.model: CLIPModel = CLIPModel.from_pretrained(model_path).to(self.device).eval()
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_path)
        self._dim = self.model.config.projection_dim

    @property
    def dim(self) -> int:
        return self._dim

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, ord=2)
        if norm < 1e-8:
            return vec
        return vec / norm

    def encode(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            feat = self.model.get_text_features(**inputs)
            arr = feat.detach().cpu().numpy()[0]
        return self._l2_normalize(arr)
