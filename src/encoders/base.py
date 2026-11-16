from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from PIL import Image


class BaseEncoder(ABC):
    """Unified encoder interface"""
    @abstractmethod
    def encode(self, input_data, normalize: bool = True) -> np.ndarray:
        raise NotImplementedError


class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, image: Image.Image) -> dict:
        """Return structured annotation dict matching video segment metadata schema"""
        raise NotImplementedError
