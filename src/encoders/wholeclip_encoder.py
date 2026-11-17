"""
CLIP4CLIP Video Sequence Encoder
Input: list of PIL frames sampled uniformly from one video segment
Internal: frame-level clip feature extract + temporal fusion
Output: single video embedding vector for the whole clip
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from src.config.settings import settings
from src.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)


class CLIP4ClipVideoEncoder(BaseEncoder):
    def __init__(self):
        self.device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        self.cache_root = Path(settings.model_cache_dir) / "clip4clip"
        self.load_path = str(self.cache_root)

        self.processor = AutoProcessor.from_pretrained(self.load_path)
        self.model = self._build_clip4clip_model()
        self.model.to(self.device)
        self.model.eval()
        logger.info("CLIP4ClipVideoEncoder loaded (temporal fusion encoder)")

    def _build_clip4clip_model(self):
        # 占位：这里替换为CLIP4CLIP官方model class
        # from clip4clip import CLIP4Clip
        # return CLIP4Clip.from_pretrained(self.load_path)
        raise NotImplementedError("Replace with official CLIP4Clip model implementation")

    @torch.no_grad()
    def encode(self, frame_list: List[Image.Image], normalize: bool = True) -> np.ndarray:
        """
        Args:
            frame_list: uniformly sampled frames from a video segment, e.g. 8/16 frames
            normalize: L2 normalize final video embedding
        Returns:
            np.ndarray: single video-level embedding
        """
        if len(frame_list) == 0:
            raise ValueError("CLIP4Clip encoder got empty frame list")

        inputs = self.processor(images=frame_list, return_tensors="pt").to(self.device)
        video_emb = self.model(**inputs)  # model output: fused temporal video feature

        if normalize:
            video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)

        return video_emb.cpu().numpy()
