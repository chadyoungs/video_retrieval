from __future__ import annotations
from typing import Literal

from src.config.settings import settings
from src.encoders.base import BaseEncoder
from src.encoders.keyframe_encoder import KeyFrameEncoder
from src.encoders.timeseries_encoder import CLIP4ClipVideoEncoder

EncoderType = Literal["keyframe", "clip4clip"]


def get_encoder(encoder_type: EncoderType) -> BaseEncoder:
    if encoder_type == "keyframe":
        return KeyFrameEncoder()
    elif encoder_type == "clip4clip":
        return CLIP4ClipVideoEncoder()
    else:
        raise ValueError(f"Unsupported encoder_type {encoder_type}")


def get_default_encoder() -> BaseEncoder:
    """Read from settings.encoder_impl: keyframe / clip4clip"""
    return get_encoder(settings.encoder_impl)
