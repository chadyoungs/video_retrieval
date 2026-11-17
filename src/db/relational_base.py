"""Relational store interface shared by every dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RelationalStoreBase(ABC):
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
    def close(self) -> None:
        """Release the engine connection pool."""
