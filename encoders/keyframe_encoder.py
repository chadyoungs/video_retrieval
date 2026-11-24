"""CLIP / SigLIP keyframe encoder.

Pipeline: read frames -> select keyframes -> embed each keyframe -> mean-pool
-> L2-normalise. Image queries reuse the same single-image path, so query and
index embeddings live in one space.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

# Import the HF env/config helper before ``transformers`` so HF_ENDPOINT is
# live when the hub reads its constants.
from config.settings import settings
from utils.hf_models import load_model, load_processor
from utils.device import resolve_device
from utils.features import features_to_array
from encoders.base import VideoEncoderBase
from encoders.keyframe_selector import select_keyframes
from encoders.video_loader import read_video_frames_raw
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


def _features_to_array(features: Any) -> np.ndarray:
    """Accept a tensor or a model output (``image_embeds`` / ``pooler_output``)."""
    return features_to_array(features)


class KeyframeEncoder(VideoEncoderBase):
    def __init__(self) -> None:
        self.model_name = settings.clip_model_name
        self.device = resolve_device()

        logger.info("Loading keyframe encoder %s on %s", self.model_name, self.device)
        self.model = load_model(self.model_name, AutoModel)
        self.processor = load_processor(self.model_name, AutoProcessor)
        self.model.to(self.device).eval()

        self.max_keyframes = settings.keyframe_count
        self._dim = self._resolve_dim()

    def _resolve_dim(self) -> int:
        config = self.model.config
        projection = getattr(config, "projection_dim", None)
        if projection:
            return int(projection)
        text_config = getattr(config, "text_config", None)
        hidden = getattr(text_config, "hidden_size", None) if text_config else None
        if hidden:
            return int(hidden)
        logger.warning(
            "Could not infer embedding dim from %s config; using settings value %d",
            self.model_name,
            settings.embedding_dim,
        )
        return settings.embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, ord=2)
        if norm < 1e-8:
            return vec
        return vec / norm

    def _encode_single_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            arr = _features_to_array(features)[0]
        return self._l2_normalize(arr)

    def encode(
        self,
        video_path: Optional[Union[str, Path]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        preloaded_frames: Optional[List[Image.Image]] = None,
    ) -> Optional[np.ndarray]:
        # Reuse frames decoded by annotation to avoid a second video pass.
        if preloaded_frames:
            raw_frames = list(preloaded_frames)
        else:
            if video_path is None or start is None or end is None:
                logger.error("Missing video path or time bounds for frame reading")
                return None
            raw_frames = read_video_frames_raw(
                video_path=video_path,
                start=start,
                end=end,
                sample_rate=settings.frame_sampling_rate,
            )
            if not raw_frames:
                logger.warning("No frames decoded: %s [%s, %s]", video_path, start, end)
                return None

        key_frames = select_keyframes(raw_frames, max_k=self.max_keyframes)
        if not key_frames:
            logger.warning("Keyframe selector returned empty set")
            return None

        frame_stack = np.stack(
            [self._encode_single_image(img) for img in key_frames], axis=0
        )
        return self._l2_normalize(np.mean(frame_stack, axis=0))

    def select_query_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        # Exactly the frames encode() embeds: same selector, same max_k.
        return select_keyframes(frames, max_k=self.max_keyframes)

    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        with Image.open(image_path) as image:
            return self._encode_single_image(image)

    # Legacy alias kept for callers from the old retrieval module.
    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        return self._encode_single_image(image)
