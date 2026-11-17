"""CLI demo for the retrieval service.

Examples
--------
Text query::

    python -m app.retrieval --type text \
        --query "busy city driving with cars and pedestrians" \
        --annotation-conditions '{"scene_env": "outdoor", "scene_type": "street"}' \
        --metadata-filters '{"camera_channel": "CAM_FRONT"}' --limit 5

Image query::

    python -m app.retrieval --type image --query ./samples/frame.png --limit 5

Video query::

    python -m app.retrieval --type video --query ./query_clip.mp4 --limit 5

Add ``--no-rerank`` to skip the integrated reranker.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Optional

from app.retrieval.service import RetrievalService


def _parse_json(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("JSON value must be an object")
        return parsed
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON argument: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid video segment retrieval")
    parser.add_argument(
        "--type",
        choices=["text", "image", "video"],
        required=True,
        help="Query modality",
    )
    parser.add_argument("--query", required=True, help="Query text, image or video path")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--dataset",
        choices=["activitynet", "nuscenes"],
        default=None,
        help="Override settings.dataset_source",
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
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the integrated reranker; return coarse hybrid order",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    annotation_conditions = _parse_json(args.annotation_conditions)
    metadata_filters = _parse_json(args.metadata_filters)

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

        for row in results:
            print(
                f"#{row['rank']} {row['clip_id']} "
                f"[{row['segment_start']}s - {row['segment_end']}s] "
                f"vec={row['vec_score']:.3f} sql={row['sql_score']:.3f} "
                f"final={row['final_score']:.3f} rerank={row['rerank_score']:.3f}"
            )
            print(f"    path: {row['video_full_path']}")
            if row.get("rerank_reason"):
                print(f"    reason: {row['rerank_reason']}")
    finally:
        service.close()


if __name__ == "__main__":
    main()
