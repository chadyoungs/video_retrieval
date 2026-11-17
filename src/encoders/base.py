from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image


class VideoEncoderBase(ABC):
    """
    Unified encoder base class.
    Only one abstract entrypoint: encode().
    Different encoder implementations define their own inner logic.
    """
    @property
    def dim(self) -> int:
        raise NotImplementedError("Subclass must implement dim property")

    @abstractmethod
    def encode(
        self,
        video_path: Optional[Union[str, Path]] = None,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        preloaded_frames: Optional[List[Image.Image]] = None,
    ) -> Optional[np.ndarray]:
        """
        Encode a video segment into embedding.
        Args:
            video_path: path to video file
            start_sec: segment start time in seconds
            end_sec: segment end time in seconds
            preloaded_frames: already decoded PIL RGB frames, skip video reading if provided
        Returns:
            normalized embedding array, None if failed / no valid frames
        """
        ...

