"""Common encoder contract shared by every representation path."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image


class VideoEncoderBase(ABC):
    """One abstract entrypoint (``encode``) plus an optional image branch.

    ``start``/``end`` are the segment bounds in seconds and match the ingestion
    call sites; do not rename them per-caller.
    """

    @property
    def dim(self) -> int:
        raise NotImplementedError("Subclass must implement dim property")

    @abstractmethod
    def encode(
        self,
        video_path: Optional[Union[str, Path]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        preloaded_frames: Optional[List["Image.Image"]] = None,
    ) -> Optional[np.ndarray]:
        """Encode a video segment into one L2-normalised embedding.

        Args:
            video_path: path to the video file.
            start: segment start in seconds.
            end: segment end in seconds.
            preloaded_frames: already decoded RGB frames; skips video reading.

        Returns:
            Normalised embedding, or ``None`` when no usable frame exists.
        """

    def encode_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Encode a single image query. Encoders without an image branch raise."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support image-query encoding"
        )
