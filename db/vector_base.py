"""Vector store interface shared by every dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStoreBase(ABC):
    @abstractmethod
    def init(self) -> None:
        """Create this dataset's own collection if missing."""

    @abstractmethod
    def drop(self) -> None:
        """Drop this dataset's own collection."""

    @abstractmethod
    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        """Insert a batch; return number of written rows."""

    @abstractmethod
    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """ANN search on this dataset's own collection."""

    @abstractmethod
    def close(self) -> None:
        """Disconnect from Milvus."""
