"""Offline checks for the ingestion pipeline wiring (no models, no DB servers).

Run from the repository root::

    python -m tests.test_ingestion_offline
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ingestion.pipeline as pipeline_module
from config.settings import settings
from annotators.rules import ANNOTATION_RULES_ACTIVITYNET
from db.models import ActivityNetSegmentRow
from tests.fakes import InMemoryRelationalStore, InMemoryVectorStore

ACTIVITYNET_COLUMNS = {column.name for column in ActivityNetSegmentRow.__table__.columns}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

ANNOTATION = {
    "scene_env": "outdoor",
    "scene_type": "street",
    "people_count": "crowd",
    "lighting": "natural_light",
    "time_of_day": "morning",
    "semantic_summary": "a busy street",
}


class FakeAnnotator:
    rules = ANNOTATION_RULES_ACTIVITYNET

    def __init__(self, annotation: Optional[dict] = None) -> None:
        self.annotation = dict(annotation or ANNOTATION)

    def annotate(self, video_path, start, end, sample_rate, preloaded_frames=None):
        return dict(self.annotation)


class FakeEncoder:
    dim = 8

    def encode(self, video_path=None, start=None, end=None, preloaded_frames=None):
        import numpy as np

        return np.arange(self.dim, dtype="float32") / 100.0


class FakeAdapter:
    def __init__(self, video_file: str) -> None:
        self.video_file = video_file

    def discover(self):
        from datasets.base import VideoEntry

        return [
            VideoEntry(
                video_file_name=Path(self.video_file).name,
                video_file_path=self.video_file,
                metadata={
                    "dataset_name": "activitynet",
                    "dataset_version": "unknown",
                    "dataset_split": "unknown",
                    "camera_channel": "unknown",
                    "extra_field_not_in_table": "should be dropped",
                },
            )
        ]


def _make_video(path: Path) -> None:
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
            str(path),
        ],
        check=True,
    )


def _install_fakes(
    video_file: str,
    rel: InMemoryRelationalStore,
    vec: InMemoryVectorStore,
    annotation: Optional[dict] = None,
) -> None:
    pipeline_module.get_default_encoder = lambda: FakeEncoder()
    pipeline_module.get_annotator = lambda dataset_name: FakeAnnotator(annotation)
    pipeline_module.get_relational_store = lambda dataset_name: rel
    pipeline_module.get_vector_store = lambda dataset_name: vec
    pipeline_module.get_dataset_adapter = lambda dataset_name: FakeAdapter(video_file)


def test_pipeline_rows_and_segments() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("skip ingestion check (opencv not installed)")
        return

    tmp_dir = Path(tempfile.mkdtemp())
    video = tmp_dir / "demo_clip.mp4"
    _make_video(video)

    original = (
        settings.clip_duration_sec,
        settings.clip_overlap_sec,
        settings.frame_sampling_rate,
        settings.batch_size_db,
    )
    settings.clip_duration_sec = 5.0
    settings.clip_overlap_sec = 0.0
    settings.frame_sampling_rate = 2
    settings.batch_size_db = 2  # force several flushes

    rel = InMemoryRelationalStore(
        rows=[], annotation_fields=tuple(ANNOTATION), columns=ACTIVITYNET_COLUMNS
    )
    vec = InMemoryVectorStore(hits=[])
    _install_fakes(str(video), rel, vec)

    try:
        pipeline = pipeline_module.IngestionPipeline(dataset_name="activitynet")
        pipeline.run()
    finally:
        (
            settings.clip_duration_sec,
            settings.clip_overlap_sec,
            settings.frame_sampling_rate,
            settings.batch_size_db,
        ) = original

    assert pipeline.stats["processed_videos"] == 1, pipeline.stats
    assert pipeline.stats["failed_videos"] == 0, pipeline.stats
    assert len(rel.rows) >= 2, f"expected multiple segments, got {len(rel.rows)}"

    first_sql, first_vec = rel.rows[0], vec.inserted[0]
    assert first_sql["segment_id"].startswith("activitynet_demo_clip_"), first_sql["segment_id"]
    assert first_sql["clip_id"] == "demo_clip"
    assert first_sql["dataset_name"] == "activitynet"
    assert first_sql["scene_type"] == "street", "annotation was not merged into the SQL row"
    assert "extra_field_not_in_table" not in first_sql, "unknown metadata leaked into the row"

    assert first_vec["segment_id"] == first_sql["segment_id"], "SQL / Milvus keys diverged"
    assert first_vec["embedding_model"] == settings.embedding_model_version
    assert len(first_vec["clip_vector"]) == FakeEncoder.dim
    assert "camera_channel" in first_vec, "Milvus row lost camera_channel"

    segment_ids = {row["segment_id"] for row in rel.rows}
    assert len(segment_ids) == len(rel.rows), "duplicate segment_id in one run"
    print(f"ok  pipeline wrote {len(rel.rows)} segments with matching keys")


def test_pipeline_limit() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("skip pipeline limit check (opencv not installed)")
        return

    tmp_dir = Path(tempfile.mkdtemp())
    video = tmp_dir / "demo_clip.mp4"
    _make_video(video)

    rel = InMemoryRelationalStore(
        rows=[], annotation_fields=tuple(ANNOTATION), columns=ACTIVITYNET_COLUMNS
    )
    vec = InMemoryVectorStore(hits=[])
    _install_fakes(str(video), rel, vec)

    pipeline = pipeline_module.IngestionPipeline(dataset_name="activitynet", limit=0)
    pipeline.run()
    assert pipeline.stats["total_videos"] == 0
    assert rel.rows == [] and vec.inserted == []
    print("ok  --limit is applied before ingestion")


def _run_quality_pass(video: Path, annotation: dict) -> tuple:
    """Ingest the demo clip once with the given annotation and return the stores."""
    rel = InMemoryRelationalStore(
        rows=[], annotation_fields=tuple(ANNOTATION), columns=ACTIVITYNET_COLUMNS
    )
    vec = InMemoryVectorStore(hits=[])
    _install_fakes(str(video), rel, vec, annotation=annotation)

    original = (settings.clip_duration_sec, settings.clip_overlap_sec, settings.frame_sampling_rate)
    settings.clip_duration_sec = 5.0
    settings.clip_overlap_sec = 0.0
    settings.frame_sampling_rate = 2
    try:
        pipeline = pipeline_module.IngestionPipeline(dataset_name="activitynet")
        pipeline.run()
    finally:
        (settings.clip_duration_sec, settings.clip_overlap_sec, settings.frame_sampling_rate) = original
    return pipeline, rel


def test_quality_flag_is_persisted() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("skip quality flag check (opencv not installed)")
        return

    tmp_dir = Path(tempfile.mkdtemp())
    video = tmp_dir / "demo_clip.mp4"
    _make_video(video)

    # Clean annotation -> rows flagged ok, no quality warnings.
    clean, clean_rel = _run_quality_pass(video, ANNOTATION)
    assert clean.stats["bad_quality_segments"] == 0, clean.stats
    assert all(row["quality_flag"] == settings.quality_flag_ok for row in clean_rel.rows), (
        clean_rel.rows[0]["quality_flag"]
    )

    # Unknown enum + meaningless summary -> rows flagged bad and counted.
    bad_annotation = dict(ANNOTATION, time_of_day="unknown", semantic_summary="???")
    dirty, dirty_rel = _run_quality_pass(video, bad_annotation)
    assert dirty.stats["bad_quality_segments"] == len(dirty_rel.rows), dirty.stats
    assert dirty.stats["quality_reasons"]["unknown:time_of_day"] == len(dirty_rel.rows)
    assert dirty.stats["quality_reasons"]["summary_no_text"] == len(dirty_rel.rows)
    assert all(
        row["quality_flag"] == settings.quality_flag_bad for row in dirty_rel.rows
    )
    assert all(row["semantic_summary"] == "???" for row in dirty_rel.rows), "bad rows are still stored"
    print(
        f"ok  quality_flag persisted ({settings.quality_flag_ok}/{settings.quality_flag_bad}) "
        f"across {len(dirty_rel.rows)} segments"
    )


def test_metadata_quality_flag_is_overridden() -> None:
    """A dataset adapter that already stamps quality_flag must not beat the assessment."""
    tmp_dir = Path(tempfile.mkdtemp())
    video = tmp_dir / "demo_clip.mp4"
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("skip quality override check (opencv not installed)")
        return
    _make_video(video)

    rel = InMemoryRelationalStore(
        rows=[], annotation_fields=tuple(ANNOTATION), columns=ACTIVITYNET_COLUMNS
    )
    vec = InMemoryVectorStore(hits=[])
    _install_fakes(
        str(video),
        rel,
        vec,
        annotation=dict(ANNOTATION, scene_type="unknown"),
    )
    original_discover = FakeAdapter.discover

    def discover_with_flag(self):
        entries = original_discover(self)
        entries[0].metadata["quality_flag"] = settings.quality_flag_ok  # stale "ok"
        return entries

    FakeAdapter.discover = discover_with_flag
    original = (settings.clip_duration_sec, settings.clip_overlap_sec, settings.frame_sampling_rate)
    settings.clip_duration_sec = 12.0
    settings.clip_overlap_sec = 0.0
    settings.frame_sampling_rate = 2
    try:
        pipeline = pipeline_module.IngestionPipeline(dataset_name="activitynet")
        pipeline.run()
    finally:
        FakeAdapter.discover = original_discover
        (settings.clip_duration_sec, settings.clip_overlap_sec, settings.frame_sampling_rate) = original

    assert rel.rows, "no rows were written"
    assert all(row["quality_flag"] == settings.quality_flag_bad for row in rel.rows), (
        "metadata quality_flag overrode the computed assessment"
    )
    print("ok  adapter metadata cannot fake the annotation assessment")


def main() -> None:
    tests = [
        test_pipeline_rows_and_segments,
        test_pipeline_limit,
        test_quality_flag_is_persisted,
        test_metadata_quality_flag_is_overridden,
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
