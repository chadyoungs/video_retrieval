"""In-memory stores and encoders for offline retrieval tests.

They implement the same interfaces as the Milvus / PostgreSQL stores so the
retrieval service can be exercised without any external service.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from db.relational_base import RelationalStoreBase
from db.vector_base import VectorStoreBase
from encoders.base import VideoEncoderBase


class InMemoryRelationalStore(RelationalStoreBase):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        annotation_fields: tuple[str, ...],
        columns: Optional[set[str]] = None,
    ) -> None:
        self.rows = rows
        self.annotation_fields = annotation_fields
        #: When set, mimic the real store's column filtering on insert.
        self.columns = columns
        self.init_called = False
        self.sql_calls: List[Dict[str, Any]] = []

    # -- lifecycle ---------------------------------------------------- #
    def init(self) -> None:
        self.init_called = True

    def drop(self) -> None:
        self.rows = []

    def close(self) -> None:
        pass

    # -- write -------------------------------------------------------- #
    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        for row in rows:
            if self.columns is not None:
                row = {key: value for key, value in row.items() if key in self.columns}
            self.rows.append(row)
        return len(rows)

    # -- read --------------------------------------------------------- #
    def search_metadata(
        self,
        annotation_conditions: Optional[Dict[str, Any]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        ann = dict(annotation_conditions or {})
        meta = dict(metadata_filters or {})
        self.sql_calls.append({"annotation": ann, "metadata": meta, "limit": limit})

        if not ann and not meta:
            return []

        hits: List[Dict[str, Any]] = []
        for row in self.rows:
            if any(str(row.get(k, "")).lower() != str(v).lower() for k, v in meta.items()):
                continue
            matched = sum(
                1 for k, v in ann.items() if str(row.get(k, "")).lower() == str(v).lower()
            )
            if ann and matched == 0:
                continue
            hit = dict(row)
            hit["score"] = matched / len(ann) if ann else 1.0
            hits.append(hit)

        hits.sort(key=lambda hit: (-hit["score"], hit["clip_id"], hit["segment_start"]))
        return hits[:limit]

    def get_segments_by_ids(self, segment_ids: List[str]) -> List[Dict[str, Any]]:
        wanted = set(segment_ids)
        return [dict(row) for row in self.rows if row["segment_id"] in wanted]


class InMemoryVectorStore(VectorStoreBase):
    dim = 8

    def __init__(self, hits: List[Dict[str, Any]], dim: int = 8) -> None:
        self.hits = hits
        self.dim = dim
        self.search_calls: List[Dict[str, Any]] = []
        self.inserted: List[Dict[str, Any]] = []

    def init(self) -> None:
        pass

    def drop(self) -> None:
        self.hits = []

    def close(self) -> None:
        pass

    def batch_insert(self, rows: List[Dict[str, Any]]) -> int:
        self.inserted.extend(rows)
        return len(rows)

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self.search_calls.append(
            {"top_k": top_k, "metadata_filters": dict(metadata_filters or {})}
        )
        hits = []
        for hit in self.hits:
            if metadata_filters and any(
                hit.get(k) != v for k, v in metadata_filters.items() if k in hit
            ):
                continue
            hits.append(dict(hit))
        return sorted(hits, key=lambda hit: -hit["score"])[:top_k]


class FakeEncoder(VideoEncoderBase):
    """Returns a fixed unit vector; counts calls instead of loading a model."""

    dim = 8

    def encode(self, video_path=None, start=None, end=None, preloaded_frames=None):
        return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

    def encode_image(self, image_path):
        return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)


class FakeTextEncoder:
    dim = 8

    def encode(self, text: str) -> np.ndarray:
        return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)
