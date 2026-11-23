"""Relational store interface shared by every dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RelationalStoreBase(ABC):
    #: Annotation fields this dataset can rank / filter on. Set by subclasses.
    annotation_fields: tuple[str, ...] = ()

    @abstractmethod
    def init(self) -> None:
        """Create this dataset's own table if missing."""

    @abstractmethod
    def drop(self) -> None:
        """Drop this dataset's own table."""

    @abstractmethod
    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        """Insert a batch; return number of written rows."""

    @abstractmethod
    def search_metadata(
        self,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Ranked metadata search on this dataset's own table."""

    @abstractmethod
    def get_segments_by_ids(self, segment_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full rows for the given ``segment_id`` values."""

    @abstractmethod
    def close(self) -> None:
        """Release the engine connection pool."""
