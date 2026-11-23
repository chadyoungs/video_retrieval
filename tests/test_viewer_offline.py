"""Offline smoke test for the retrieval result viewer (headless, Agg backend).

Run from the repository root::

    python -m tests.test_viewer_offline
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
            "-t", "8", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def main() -> None:
    try:
        import matplotlib  # noqa: F401
        import cv2  # noqa: F401
    except ImportError as exc:
        print(f"skip viewer check ({exc})")
        return

    from retrieval.viewer import show_results

    tmp_dir = Path(tempfile.mkdtemp())
    video = tmp_dir / "demo.mp4"
    _make_video(video)

    results = [
        {
            "rank": 1,
            "segment_id": "activitynet_demo_0_5000",
            "clip_id": "demo",
            "video_full_path": str(video),
            "segment_start": 0.0,
            "segment_end": 5.0,
            "quality_flag": "ok",
            "vec_score": 0.91,
            "sql_score": 0.5,
            "coarse_score": 0.79,
            "rerank_score": 0.83,
            "score_rule": 0.5,
            "score_cross_encoder": None,
            "score_vlm": 0.9,
            "rerank_reason": "rule+vlm",
            "annotation": {
                "scene_env": "outdoor",
                "semantic_summary": "a busy street with pedestrians",
            },
        },
        {
            "rank": 2,
            "segment_id": "activitynet_missing_0_5000",
            "clip_id": "missing",
            "video_full_path": str(tmp_dir / "does_not_exist.mp4"),
            "segment_start": 0.0,
            "segment_end": 5.0,
            "quality_flag": "bad",
            "vec_score": 0.7,
            "sql_score": 0.0,
            "coarse_score": 0.49,
            "rerank_score": 0.49,
            "annotation": {"semantic_summary": "unknown"},
        },
    ]

    save_path = show_results(
        results, query="busy street", frames_per_segment=3,
        save_dir=str(tmp_dir / "views"),
    )
    assert save_path is not None and save_path.exists(), "viewer did not save a PNG"
    assert save_path.stat().st_size > 20_000, "PNG looks empty"

    assert show_results([], save_dir=str(tmp_dir)) is None
    print(f"ok  viewer rendered {len(results)} hits -> {save_path.name}")
    print("\n1/1 checks passed")


if __name__ == "__main__":
    main()
