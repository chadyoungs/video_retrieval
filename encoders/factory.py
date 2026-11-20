from __future__ import annotations
from typing import Literal

from config.settings import settings
from encoders.base import VideoEncoderBase
from encoders.keyframe_encoder import KeyframeEncoder
from encoders.text_encoder import TextEncoder
from encoders.wholeclip_encoder import CLIP4ClipVideoEncoder

EncoderType = Literal["keyframes", "wholeclip"]


def get_encoder(encoder_type: EncoderType) -> VideoEncoderBase:
    if encoder_type == "keyframes":
        return KeyframeEncoder()
    elif encoder_type == "wholeclip":
        return CLIP4ClipVideoEncoder()
    else:
        raise ValueError(f"Unsupported encoder_type {encoder_type}")

def get_text_encoder() -> TextEncoder:
    """Get standalone text encoder for text query retrieval"""
    return TextEncoder()

def get_default_encoder() -> VideoEncoderBase:
    """Read from settings.encoder_type: keyframes / wholeclip"""
    return get_encoder(settings.encoder_type)
