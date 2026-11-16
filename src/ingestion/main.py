import argparse
import logging

from src.ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser("Video ingestion pipeline")
    parser.add_argument("--dataset", type=str, default="activitynet", help="Dataset name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos for test")
    parser.add_argument("--drop-sql", action="store_true", help="Drop sql table before run")
    parser.add_argument("--drop-milvus", action="store_true", help="Drop milvus collection before run")
    args = parser.parse_args()

    pipeline = IngestionPipeline(
        dataset_name=args.dataset,
        drop_sql=args.drop_sql,
        drop_milvus=args.drop_milvus,
        limit=args.limit,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
