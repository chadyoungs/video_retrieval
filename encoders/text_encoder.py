"""Standalone text-query encoder.

It loads the *same* model family as the video encoder (CLIP or SigLIP, chosen by
``settings.semantic_model_tier``) so text and clip vectors share one space and
one dimension. Loading a different model here silently breaks ANN search.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

# Import the HF env/config helper before ``transformers`` so HF_ENDPOINT is
# live when the hub reads its constants.
from config.settings import settings
from utils.hf_models import load_model, load_processor
from utils.device import resolve_device
from utils.features import features_to_array
from transformers import AutoModel, AutoProcessor

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

        # Surface the concrete type when someone debugs a shape mismatch.
        logger.debug("Text feature output type: %s", type(features).__name__)
        # Newer transformers / some checkpoint families return a model output
        # object instead of a tensor, so normalise before touching ``detach``.
        arr = features_to_array(features)[0]

        if arr.shape[-1] != self._dim:
            raise ValueError(
                f"Text encoder {self.model_name} produced {arr.shape[-1]}-d features "
                f"but the index expects {self._dim}-d. The checkpoint exposed pooled "
                "hidden states instead of the projected text embedding (e.g. a "
                "text-only CLIPTextModel). Load the multimodal checkpoint "
                "(openai/clip-vit-base-patch32, google/siglip-so400m-patch14-384) "
                "and keep SEMANTIC_MODEL_TIER aligned with the index."
            )
        return self._l2_normalize(arr)
