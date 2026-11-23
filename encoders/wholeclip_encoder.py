"""CLIP4Clip sequence encoder (not wired up yet).

Kept as a clearly-marked stub: ``settings.encoder_type = "whole_clip"`` must fail
at construction time, not produce silently wrong clips. The interface matches
:class:`encoders.base.VideoEncoderBase` so the switch is a drop-in once the
official CLIP4Clip checkpoint/class is installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from config.settings import settings
from encoders.base import VideoEncoderBase
from utils.device import resolve_device
from utils.hf_models import local_model_dir

logger = logging.getLogger(__name__)


class CLIP4ClipVideoEncoder(VideoEncoderBase):
    def __init__(self) -> None:
        self.device = resolve_device()
        self.load_path = local_model_dir("clip4clip")
        if not self.load_path.exists():
            raise FileNotFoundError(
                f"CLIP4Clip weights not found at {self.load_path}. Either place the "
                f"official checkpoint there or keep encoder_type='keyframes'."
            )
        self.model = self._build_clip4clip_model().to(self.device).eval()
        self._dim = int(getattr(self.model.config, "projection_dim", settings.embedding_dim))
        logger.info("CLIP4ClipVideoEncoder loaded on %s", self.device)

    def _build_clip4clip_model(self):
        # Replace with the official CLIP4Clip model class:
        #   from clip4clip import CLIP4Clip
        #   return CLIP4Clip.from_pretrained(self.load_path)
        raise NotImplementedError(
            "CLIP4Clip model class is not implemented; use encoder_type='keyframes'"
        )

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(
        self,
        video_path: Optional[Union[str, Path]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        preloaded_frames: Optional[List[object]] = None,
    ) -> Optional[np.ndarray]:
        frame_list = preloaded_frames
        if not frame_list:
            if video_path is None or start is None or end is None:
                logger.error("Missing video path or segment bounds for CLIP4Clip")
                return None
            from encoders.video_loader import read_video_frames_raw

            frame_list = read_video_frames_raw(
                video_path=video_path,
                start=start,
                end=end,
                sample_rate=settings.frame_sampling_rate,
            )
        if not frame_list:
            return None

        inputs = self.processor(images=frame_list, return_tensors="pt").to(self.device)
        video_emb = self.model(**inputs)
        if isinstance(video_emb, dict):
            video_emb = video_emb["video_emb"]
        video_emb = video_emb / video_emb.norm(dim=-1, keepdim=True)
        return video_emb.cpu().numpy()[0]
