"""CLI: run the ingestion pipeline for one dataset.

Examples
--------
    python -m ingestion.main --dataset activitynet --limit 20
    python -m ingestion.main --dataset activitynet --drop-sql --drop-milvus
"""

import argparse
import logging
import os

from config.settings import settings
from ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser("Video ingestion pipeline")
    parser.add_argument(
        "--dataset",
        default=settings.dataset_source,
        choices=["activitynet", "nuscenes"],
        help="Dataset key (also selects the PG table and Milvus collection)",
    )
    # We can run a small dataset for quick validation.
    parser.add_argument("--limit", type=int, default=None, help="Only ingest the first N videos")
    parser.add_argument("--drop-sql", action="store_true", help="Drop the PG table first")
    parser.add_argument("--drop-milvus", action="store_true", help="Drop the Milvus collection first")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if settings.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

    if args.dataset == "activitynet" and not os.path.isdir(settings.activitynet_data_root):
        raise SystemExit(
            f"activitynet_data_root does not exist: {settings.activitynet_data_root}. "
            f"Set ACTIVITYNET_DATA_ROOT or edit .env."
        )
    if args.dataset == "nuscenes" and not os.path.isfile(settings.nuscenes_clip_mapping_file):
        raise SystemExit(
            f"nuScenes clip mapping not found: {settings.nuscenes_clip_mapping_file}. "
            f"Run datasets/builders/nuscenes_video_builder.py first."
        )

    pipeline = IngestionPipeline(
        dataset_name=args.dataset,
        drop_sql=args.drop_sql,
        drop_milvus=args.drop_milvus,
        limit=args.limit,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
