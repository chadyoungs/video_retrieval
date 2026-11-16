"""Abstract interface for the vector store.
No SQL / relational knowledge here.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStoreBase(ABC):
    @abstractmethod
    def create_collection(self, drop_existing: bool = False) -> None:
        """Create milvus collection and index for this dataset."""

    @abstractmethod
    def insert_vectors(self, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Insert records with embedding vector."""

    @abstractmethod
    def search_vectors(
        self,
        query_embedding: List[float],
        limit: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Vector similarity search, return hits with normalized score [0,1]."""

    @abstractmethod
    def close(self) -> None:
        """Close milvus client."""
