"""Encoder factories.

Heavy model imports are deferred to instantiation so that importing this module
(and therefore the retrieval service) does not require torch / transformers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional

from config.settings import settings
from encoders.base import VideoEncoderBase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from encoders.text_encoder import TextEncoder

logger = logging.getLogger(__name__)

#: Accepted spellings: settings uses "whole_clip", the legacy CLI used
#: "wholeclip". Both resolve to the CLIP4Clip encoder.
EncoderType = Literal["keyframes", "whole_clip", "wholeclip"]

_ENCODER_ALIASES = {
    "keyframes": "keyframes",
    "whole_clip": "whole_clip",
    "wholeclip": "whole_clip",
}


def get_encoder(encoder_type: Optional[EncoderType] = None) -> VideoEncoderBase:
    """Build the encoder for ``encoder_type`` (default: settings.encoder_type)."""
    key = encoder_type or settings.encoder_type
    normalized = _ENCODER_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported encoder_type {key!r}; expected one of "
            f"{sorted(_ENCODER_ALIASES)}"
        )

    if normalized == "keyframes":
        from encoders.keyframe_encoder import KeyframeEncoder

        return KeyframeEncoder()

    from encoders.wholeclip_encoder import CLIP4ClipVideoEncoder

    return CLIP4ClipVideoEncoder()


def get_text_encoder() -> "TextEncoder":
    """Standalone text encoder for text-query retrieval."""
    from encoders.text_encoder import TextEncoder

    return TextEncoder()


def get_default_encoder() -> VideoEncoderBase:
    """Encoder selected by ``settings.encoder_type``."""
    return get_encoder(settings.encoder_type)
