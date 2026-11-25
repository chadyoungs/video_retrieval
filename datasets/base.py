"""Common contract for dataset adapters.

An adapter discovers source videos (or pre-built segments) for one dataset and
returns a uniform :class:`VideoEntry` list. It has no dependency on the
encoder, stores or retrieval layer; ingestion code only sees this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VideoEntry(BaseModel):
    """One source video or pre-built segment handed to the ingestion pipeline."""

    video_file_name: str
    video_file_path: str
    # None for whole-video ActivityNet entries; set for NuScenes segments.
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetAdapterBase(ABC):
    """Per-dataset discovery and metadata normalization contract.

    Two dataset traits drive ingestion and query encoding, so every consumer
    reads them from the adapter instead of hard-coding dataset names:

    * ``presegmented`` - each discovered file is already one final segment
      (nuScenes builder clips); no sliding window, no overlap.
    * ``frame_sampling_rate`` - decode fps for annotation / embedding.
      Continuous video (ActivityNet) is resampled; low-rate keyframe data
      (nuScenes, 2 Hz = two images per second) is taken frame-complete.
    """

    #: True when one discovered file equals one final segment.
    presegmented: bool = False

    @property
    def frame_sampling_rate(self) -> float:
        """Frames per second to decode from source files."""
        from config.settings import settings

        return float(settings.frame_sampling_rate)

    @abstractmethod
    def discover(self) -> List[VideoEntry]:
        """Return all available source entries for this dataset."""
