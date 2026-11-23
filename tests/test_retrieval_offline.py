"""Offline checks for the retrieval path (no Milvus / PostgreSQL / CLIP / Ollama).

Run from the repository root::

    python -m tests.test_retrieval_offline

Both external stores and the model stack are replaced by in-memory fakes; the
real ``RetrievalService``, ``IntegratedReranker`` and fusion logic run unchanged.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from rerankers.integrated import IntegratedReranker
from retrieval.service import RetrievalService
from tests.fakes import FakeEncoder, FakeTextEncoder, InMemoryRelationalStore, InMemoryVectorStore

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

VIDEO_DIR = "/tmp/video_retrieval_demo"

ROWS: List[Dict[str, Any]] = [
    {
        "segment_id": "activitynet_a_0_10000",
        "clip_id": "a",
        "video_file_name": "a.mp4",
        "video_file_path": VIDEO_DIR,
        "segment_start": 0.0,
        "segment_end": 10.0,
        "dataset_name": "activitynet",
        "camera_channel": "CAM_FRONT",
        "scene_env": "outdoor",
        "scene_type": "street",
        "people_count": "crowd",
        "lighting": "natural_light",
        "time_of_day": "morning",
        "semantic_summary": "a busy street with many pedestrians",
    },
    {
        "segment_id": "activitynet_a_8000_18000",
        "clip_id": "a",
        "video_file_name": "a.mp4",
        "video_file_path": VIDEO_DIR,
        "segment_start": 8.0,
        "segment_end": 18.0,
        "dataset_name": "activitynet",
        "camera_channel": "CAM_BACK",
        "scene_env": "indoor",
        "scene_type": "office",
        "people_count": "single",
        "lighting": "dim",
        "time_of_day": "noon",
        "semantic_summary": "one person working in a dim office",
    },
    {
        "segment_id": "activitynet_b_0_10000",
        "clip_id": "b",
        "video_file_name": "b.mp4",
        "video_file_path": VIDEO_DIR,
        "segment_start": 0.0,
        "segment_end": 10.0,
        "dataset_name": "activitynet",
        "camera_channel": "CAM_FRONT",
        "scene_env": "outdoor",
        "scene_type": "park",
        "people_count": "few",
        "lighting": "natural_light",
        "time_of_day": "morning",
        "semantic_summary": "people walking through a park",
    },
]

# Vector scores are already normalised to [0, 1] by the store adapter.
VEC_HITS: List[Dict[str, Any]] = [
    {
        "segment_id": "activitynet_a_0_10000",
        "clip_id": "a",
        "video_file_name": "a.mp4",
        "video_file_path": VIDEO_DIR,
        "segment_start": 0.0,
        "segment_end": 10.0,
        "dataset_name": "activitynet",
        "camera_channel": "CAM_FRONT",
        "embedding_model": "clip",
        "score": 0.90,
        "distance": 0.80,
    },
    {
        "segment_id": "activitynet_a_8000_18000",
        "clip_id": "a",
        "video_file_name": "a.mp4",
        "video_file_path": VIDEO_DIR,
        "segment_start": 8.0,
        "segment_end": 18.0,
        "dataset_name": "activitynet",
        "camera_channel": "CAM_BACK",
        "embedding_model": "clip",
        "score": 0.80,
        "distance": 0.60,
    },
    {
        "segment_id": "activitynet_b_0_10000",
        "clip_id": "b",
        "video_file_name": "b.mp4",
        "video_file_path": VIDEO_DIR,
        "segment_start": 0.0,
        "segment_end": 10.0,
        "dataset_name": "activitynet",
        "camera_channel": "CAM_FRONT",
        "embedding_model": "clip",
        "score": 0.70,
        "distance": 0.40,
    },
]


def build_service(enable_ce: bool = False, enable_vlm: bool = False) -> RetrievalService:
    # Default: rule-only reranking, so no model weights and no Ollama endpoint
    # are needed. The degradation tests switch the optional components back on.
    settings.reranker_enable_cross_encoder = enable_ce
    settings.reranker_enable_vlm = enable_vlm

    rel = InMemoryRelationalStore(rows=[dict(row) for row in ROWS], annotation_fields=(
        "scene_env",
        "scene_type",
        "people_count",
        "lighting",
        "time_of_day",
        "semantic_summary",
    ))
    vec = InMemoryVectorStore(hits=[dict(hit) for hit in VEC_HITS])
    reranker = IntegratedReranker("activitynet")
    return RetrievalService(
        dataset_key="activitynet",
        relational_store=rel,
        vector_store=vec,
        reranker=reranker,
        encoder=FakeEncoder(),
        text_encoder=FakeTextEncoder(),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_segments_stay_distinct() -> None:
    """Two segments of the same clip must keep their own scores and rank slots."""
    service = build_service()
    results = service.search_text("people outside", rerank=False, limit=3)

    ids = [row["segment_id"] for row in results]
    assert len(ids) == len(set(ids)), f"duplicate segments in output: {ids}"
    assert "activitynet_a_0_10000" in ids and "activitynet_a_8000_18000" in ids

    scores = {row["segment_id"]: row["vec_score"] for row in results}
    assert scores["activitynet_a_0_10000"] == 0.90
    assert scores["activitynet_a_8000_18000"] == 0.80, (
        "the same clip's second segment inherited the wrong vector score"
    )
    print("ok  distinct segments keep their own vectors")


def test_no_rerank_output_contract() -> None:
    service = build_service()
    results = service.search_text("people outside", rerank=False, limit=2)
    assert len(results) == 2
    required = {
        "rank",
        "segment_id",
        "clip_id",
        "video_full_path",
        "vec_score",
        "sql_score",
        "coarse_score",
        "final_score",
        "rerank_score",
        "rerank_reason",
    }
    assert required.issubset(results[0]), required - set(results[0])
    assert results[0]["video_full_path"].endswith(".mp4")
    print("ok  coarse output contract")


def test_annotation_backfill() -> None:
    """Vector-only candidates get annotation text from PostgreSQL."""
    service = build_service()
    results = service.search_text("people outside", rerank=True, limit=3)
    annotated = [row for row in results if row["annotation"]]
    assert annotated, "no candidate was enriched with annotation"
    assert annotated[0]["annotation"]["scene_env"] in {"outdoor", "indoor"}
    print("ok  annotation back-filled for vector-only candidates")


def test_rule_scoring_orders_candidates() -> None:
    service = build_service()
    results = service.search_text(
        "people outside",
        annotation_conditions={"scene_env": "outdoor"},
        rerank=True,
        limit=3,
    )
    assert results, "rule-only rerank returned nothing"
    first_two = [row["annotation"]["scene_env"] for row in results[:2]]
    assert first_two == ["outdoor", "outdoor"], results
    assert results[0]["score_rule"] == 1.0
    assert "rule=" in results[0]["rerank_reason"]
    print("ok  rule score ranks annotation matches first")


def test_metadata_filter_is_hard() -> None:
    """Fields stored only in PostgreSQL are still enforced after fusion.

    ``scene_type`` is an annotation column, so Milvus cannot filter on it; the
    fake vector store returns every hit, exactly like an unfiltered ANN call.
    """
    service = build_service()
    results = service.search_text(
        "people outside",
        metadata_filters={"scene_type": "street"},
        rerank=False,
        limit=5,
    )
    ids = [row["segment_id"] for row in results]
    assert ids == ["activitynet_a_0_10000"], ids
    assert results[0]["annotation"]["scene_type"] == "street"
    print("ok  hard metadata filter drops non-matching segments")


def test_reranker_gate_returns_empty() -> None:
    service = build_service()
    results = service.search_text(
        "people outside",
        metadata_filters={"scene_type": "kitchen"},
        rerank=True,
        limit=5,
    )
    assert results == [], results
    print("ok  impossible filter yields no results")


def test_reranker_rejects_all_when_no_signal() -> None:
    """With CE/VLM off and no annotation conditions, coarse order is kept."""
    service = build_service()
    results = service.search_text("people outside", rerank=True, limit=3)
    vec_order = [row["segment_id"] for row in sorted(
        VEC_HITS, key=lambda hit: -hit["score"]
    )]
    assert [row["segment_id"] for row in results] == vec_order
    assert "coarse" in results[0]["rerank_reason"]
    print("ok  no-signal rerank falls back to coarse order")


def test_sql_candidates_enter_recall() -> None:
    """Annotation-filtered rows missing from the ANN top-k still get recalled."""
    service = build_service()
    rel = service.relational_store
    rel.rows.append(
        {
            "segment_id": "activitynet_c_0_10000",
            "clip_id": "c",
            "video_file_name": "c.mp4",
            "video_file_path": VIDEO_DIR,
            "segment_start": 0.0,
            "segment_end": 10.0,
            "dataset_name": "activitynet",
            "camera_channel": "CAM_FRONT",
            "scene_env": "outdoor",
            "scene_type": "field",
            "people_count": "few",
            "lighting": "natural_light",
            "time_of_day": "morning",
            "semantic_summary": "an empty field",
        }
    )
    results = service.search_text(
        "empty field",
        annotation_conditions={"scene_type": "field"},
        rerank=False,
        limit=5,
    )
    assert "activitynet_c_0_10000" in [row["segment_id"] for row in results]
    print("ok  annotation-filtered SQL candidates enter recall")


def test_postgres_store_sql_generation() -> None:
    """The real SQLAlchemy store builds valid ranked queries (SQLite backend)."""
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        print("skip postgres store check (sqlalchemy not installed)")
        return

    import tempfile

    from config.settings import settings
    from db.sql_activitynet import ActivityNetRelationalStore

    original_dsn = settings.pg_dsn
    settings.pg_dsn = f"sqlite:///{Path(tempfile.mkdtemp()) / 'segments.db'}"
    store = ActivityNetRelationalStore()
    try:
        store.init()
        assert store.batch_insert([dict(row) for row in ROWS]) == len(ROWS)

        ranked = store.search_metadata(
            annotation_conditions={"scene_env": "outdoor", "people_count": "few"}, limit=5
        )
        assert ranked, "annotation ranking returned nothing"
        assert ranked[0]["segment_id"] == "activitynet_b_0_10000"
        assert ranked[0]["score"] == 1.0, "both conditions matched"
        assert ranked[1]["segment_id"] == "activitynet_a_0_10000"
        assert ranked[1]["score"] == 0.5, "one of two conditions matched"

        filtered = store.search_metadata(metadata_filters={"scene_type": "street"}, limit=5)
        assert {row["segment_id"] for row in filtered} == {"activitynet_a_0_10000"}

        empty = store.search_metadata()
        assert empty == [], "unscored full-table scan must not be used as recall"

        lookup = store.get_segments_by_ids(["activitynet_a_0_10000", "missing"])
        assert len(lookup) == 1 and lookup[0]["scene_type"] == "street"
    finally:
        store.close()
        settings.pg_dsn = original_dsn
    print("ok  postgres store SQL: ranking, filters, segment lookup")


def test_optional_components_degrade_gracefully() -> None:
    """CE weights / Ollama missing must not break reranking or pad scores."""
    service = build_service(enable_ce=True, enable_vlm=True)
    results = service.search_text(
        "people outside",
        annotation_conditions={"scene_env": "outdoor"},
        rerank=True,
        limit=3,
    )
    assert results, "rerank returned nothing when optional components are down"
    assert results[0]["score_rule"] == 1.0
    assert results[0]["score_cross_encoder"] is None, results[0]
    assert results[0]["score_vlm"] is None, results[0]
    # Only the rule weight contributed, so the fused score must equal the rule
    # score rather than be diluted to 0.4 * rule.
    assert abs(results[0]["rerank_score"] - 1.0) < 1e-6, results[0]["rerank_score"]
    print("ok  missing CE / VLM are skipped and weights renormalized")


def test_video_loader_sampling() -> None:
    """Frame sampling uses real timestamps and honours the requested window."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("skip video loader check (opencv not installed)")
        return

    import subprocess
    import tempfile

    from encoders.video_loader import read_video_frames_raw

    tmp = Path(tempfile.mkdtemp()) / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=30",
            "-t",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(tmp),
        ],
        check=True,
    )

    frames = read_video_frames_raw(tmp, start=4.0, end=6.0, sample_rate=5)
    assert 8 <= len(frames) <= 12, f"unexpected frame count {len(frames)}"
    frames_full = read_video_frames_raw(tmp, start=0.0, end=12.0, sample_rate=2)
    assert 20 <= len(frames_full) <= 26, f"unexpected full-clip frame count {len(frames_full)}"
    print("ok  video loader seeks and samples by timestamp")


def main() -> None:
    tests = [
        test_segments_stay_distinct,
        test_no_rerank_output_contract,
        test_annotation_backfill,
        test_rule_scoring_orders_candidates,
        test_metadata_filter_is_hard,
        test_reranker_gate_returns_empty,
        test_reranker_rejects_all_when_no_signal,
        test_sql_candidates_enter_recall,
        test_postgres_store_sql_generation,
        test_optional_components_degrade_gracefully,
        test_video_loader_sampling,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failed}/{len(tests)} checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
