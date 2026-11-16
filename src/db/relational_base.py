"""Abstract interface for the relational metadata store.
No Milvus / vector knowledge here.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RelationalStoreBase(ABC):
    @abstractmethod
    def init_schema(self, drop_existing: bool = False) -> None:
        """Create tables and indexes for this dataset."""

    @abstractmethod
    def insert_records(self, records: List[Dict[str, Any]]) -> int:
        """Batch insert metadata records, return inserted row count."""

    @abstractmethod
    def search_metadata(
        self,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Metadata filter + annotation OR match ranking, return scored hits."""

    @abstractmethod
    def close(self) -> None:
        """Close DB connection."""
