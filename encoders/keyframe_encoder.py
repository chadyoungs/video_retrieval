from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config.settings import settings
from src.encoders.base import VideoEncoderBase
from src.encoders.keyframe_selector import select_keyframes
from src.encoders.video_loader import read_video_frames_raw

logger = logging.getLogger(__name__)


class KeyframeEncoder(VideoEncoderBase):
    """
    Pipeline:
    1. Read frames (or reuse preloaded_frames)
    2. Select keyframes by keyframe selector
    3. Encode each keyframe image embedding
    4. Mean pooling over keyframe embeddings
    5. L2 normalize final aggregated vector
    """
    def __init__(
        self
    ):
        self.model_name = settings.clip_model_name
        self.device = settings.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_keyframes = settings.keyframe_count

        logger.info(f"Loading CLIP KeyframeEncoder: {self.model_name} on {self.device}")
        self.model: CLIPModel = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self.processor: CLIPProcessor = CLIPProcessor.from_pretrained(self.model_name)
        self._dim = self.model.config.projection_dim

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_single_image(self, image: Image.Image) -> np.ndarray:
        """Internal helper: encode one single image"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            feat = self.model.get_image_features(**inputs)
            arr = feat.detach().cpu().numpy()[0]
        return arr

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, ord=2)
        if norm < 1e-8:
            return vec
        return vec / norm

    def encode(
        self,
        video_path: Optional[Union[str, Path]] = None,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        preloaded_frames: Optional[List[Image.Image]] = None,
    ) -> Optional[np.ndarray]:
        # use preloaded frames first to avoid repeated decoding
        if preloaded_frames is not None and len(preloaded_frames) > 0:
            raw_frames = preloaded_frames
        else:
            if video_path is None or start_sec is None or end_sec is None:
                logger.error("Missing video path or time bounds for frame reading")
                return None
            raw_frames = read_video_frames_raw(
                video_path=video_path,
                start=start_sec,
                end=end_sec,
                sample_rate=settings.frame_sampling_rate
            )
            if not raw_frames:
                logger.warning(f"No frames decoded: {video_path} [{start_sec}, {end_sec}]")
                return None

        # run keyframe selection
        key_frames = select_keyframes(raw_frames, max_k=self.max_keyframes)
        if not key_frames:
            logger.warning("Keyframe selector returned empty set")
            return None

        # encode every keyframe
        frame_embeds = []
        for img in key_frames:
            emb = self._encode_single_image(img)
            frame_embeds.append(emb)
        frame_stack = np.stack(frame_embeds, axis=0)

        # mean pooling across keyframes
        pooled = np.mean(frame_stack, axis=0)
        final_emb = self._l2_normalize(pooled)
        return final_emb

    def encode_image_query(self, image: Image.Image) -> np.ndarray:
        return self._l2_normalize(self._encode_single_image(image))
