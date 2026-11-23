"""CLI for the retrieval service.

Examples
--------
Text query::

    python -m retrieval.main --type text --query "a man playing guitar" --limit 5
    python -m retrieval.main --type text --query "outdoor street with cars" \
        --annotation-conditions '{"scene_env": "outdoor"}' \
        --metadata-filters '{"dataset_name": "activitynet"}'

Image query::

    python -m retrieval.main --type image --query ./samples/frame.png --limit 5

Video query::

    python -m retrieval.main --type video --query ./query_clip.mp4 --limit 5

Add ``--no-rerank`` to inspect the coarse hybrid order and ``--json`` for
machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Optional

from config.settings import settings
from retrieval.service import RetrievalService


def _parse_json(value: Optional[str], name: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for {name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _print_row(row: Dict[str, Any]) -> None:
    print(
        f"#{row['rank']} {row['segment_id']} [{row['segment_start']}s - {row['segment_end']}s] "
        f"vec={row['vec_score']:.3f} sql={row['sql_score']:.3f} "
        f"coarse={row['coarse_score']:.3f} final={row['rerank_score']:.3f}"
    )
    print(f"    clip: {row['clip_id']} quality: {row.get('quality_flag')}")
    print(f"    path: {row['video_full_path']}")
    if row.get("annotation"):
        print(f"    annotation: {row['annotation']}")
    if row.get("rerank_reason"):
        print(f"    reason: {row['rerank_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid video segment retrieval")
    parser.add_argument("--type", choices=["text", "image", "video"], required=True)
    parser.add_argument("--query", required=True, help="Query text, image path or video path")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--dataset",
        choices=["activitynet", "nuscenes"],
        default=None,
        help="Dataset key; defaults to settings.dataset_source",
    )
    parser.add_argument(
        "--annotation-conditions",
        default="{}",
        help='Annotation OR conditions as JSON, e.g. \'{"scene_env": "outdoor"}\'',
    )
    parser.add_argument(
        "--metadata-filters",
        default="{}",
        help='Metadata AND filters as JSON, e.g. \'{"camera_channel": "CAM_FRONT"}\'',
    )
    parser.add_argument("--no-rerank", action="store_true", help="Return the coarse hybrid order")
    parser.add_argument(
        "--only-good",
        action="store_true",
        help="Shortcut for metadata filter quality_flag=<ok flag> (see QUALITY_FLAG_OK)",
    )
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    annotation_conditions = _parse_json(args.annotation_conditions, "--annotation-conditions")
    metadata_filters = _parse_json(args.metadata_filters, "--metadata-filters")
    if args.only_good:
        metadata_filters.setdefault("quality_flag", settings.quality_flag_ok)

    service = RetrievalService(dataset_key=args.dataset)
    try:
        if args.type == "text":
            results = service.search_text(
                query_text=args.query,
                annotation_conditions=annotation_conditions,
                metadata_filters=metadata_filters,
                limit=args.limit,
                rerank=not args.no_rerank,
            )
        elif args.type == "image":
            results = service.search_image(
                query_image_path=args.query,
                annotation_conditions=annotation_conditions,
                metadata_filters=metadata_filters,
                limit=args.limit,
                rerank=not args.no_rerank,
            )
        else:
            results = service.search_video(
                query_video_path=args.query,
                annotation_conditions=annotation_conditions,
                metadata_filters=metadata_filters,
                limit=args.limit,
                rerank=not args.no_rerank,
            )
    finally:
        service.close()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        return
    if not results:
        print("No matches.")
        return
    for row in results:
        _print_row(row)


if __name__ == "__main__":
    main()
