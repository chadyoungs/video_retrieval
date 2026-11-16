from __future__ import annotations

from typing import Literal

from src.rerankers.integrated import IntegratedReranker
from src.rerankers.base import BaseReranker
from src.config.settings import settings
from typing import Optional, Literal

DatasetKey = Literal["activitynet", "nuscenes"]

def get_reranker(dataset_key: Optional[DatasetKey] = None) -> BaseReranker:
    key = dataset_key or settings.dataset_source
    return IntegratedReranker(dataset_key=key)

