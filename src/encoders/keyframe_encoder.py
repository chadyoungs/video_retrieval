"""
KeyFrame Encoder: single image embedding for sampled keyframe
Support CLIP / SigLIP, same as your original workflow:
video segment -> extract one keyframe -> this encoder -> embedding
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.config.settings import settings
from src.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)


class KeyFrameEncoder(BaseEncoder):
    def __init__(self):
        self.device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        self.cache_root = Path(settings.model_cache_dir)
        self.preset = settings.encoder_type
        self.local_path = self.cache_root / self.preset

        PRESET_MAP = {
            "clip": "openai/clip-vit-base-patch32",
            "siglip": "google/siglip-so400m-patch14-384",
        }
        hf_repo = PRESET_MAP[self.preset]

        if self.local_path.exists() and any(self.local_path.iterdir()):
            load_dir = str(self.local_path)
            logger.info(f"KeyFrameEncoder load local {self.preset}: {load_dir}")
        else:
            load_dir = hf_repo
            logger.warning(f"KeyFrameEncoder load from HF hub: {hf_repo}")

        self.processor = AutoProcessor.from_pretrained(load_dir)
        self.model = AutoModel.from_pretrained(load_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, image: Union[Image.Image, List[Image.Image]], normalize: bool = True) -> np.ndarray:
        is_single = isinstance(image, Image.Image)
        if is_single:
            image = [image]

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        emb = out.image_embeds

        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)

        arr = emb.cpu().numpy()
        return arr[0] if is_single else arr
