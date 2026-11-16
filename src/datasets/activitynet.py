"""ActivityNet adapter: glob raw videos under the configured data root.

Digest of ``utils/dataset_adapter.py::_build_activitynet_entries``. ActivityNet
entries are whole videos; segment bounds are left unset and resolved by the
ingestion step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.config.settings import settings
from src.datasets.base import DatasetAdapterBase, VideoEntry

logger = logging.getLogger(__name__)


class ActivityNetAdapter(DatasetAdapterBase):
    """Discover ActivityNet videos matching the configured file format."""

    def discover(self) -> List[VideoEntry]:
        root = Path(settings.activitynet_root)
        if not root.exists():
            logger.warning("ActivityNet data root does not exist: %s", root)
            return []

        pattern = f"*.{settings.activitynet_file_format}"
        entries: List[VideoEntry] = []
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            entries.append(
                VideoEntry(
                    video_file_name=path.name,
                    video_file_path=str(path),
                    metadata={
                        "dataset_name": "activitynet",
                        "dataset_version": "unknown",
                        "dataset_split": "unknown",
                        "camera_channel": "unknown",
                    },
                )
            )

        logger.info("Discovered %d ActivityNet videos under %s", len(entries), root)
        return entries
