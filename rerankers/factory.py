from __future__ import annotations

from typing import Literal

from rerankers.integrated import IntegratedReranker
from rerankers.base import BaseReranker
from config.settings import settings
from typing import Optional, Literal

DatasetName = Literal["activitynet", "nuscenes"]

def get_reranker(dataset_name: Optional[DatasetName] = None) -> BaseReranker:
    name = dataset_name or settings.dataset_source
    return IntegratedReranker(dataset_name=name)

