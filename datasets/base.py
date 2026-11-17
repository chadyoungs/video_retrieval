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
    """Per-dataset discovery and metadata normalization contract."""

    @abstractmethod
    def discover(self) -> List[VideoEntry]:
        """Return all available source entries for this dataset."""
