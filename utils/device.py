"""torch device resolution shared by encoders and rerankers.

``settings.device`` accepts ``auto`` / ``cpu`` / ``cuda`` / ``cuda:0`` / ``mps``.
``auto`` prefers CUDA, then Apple MPS, else CPU, so the same code runs on the
GPU box and on a laptop.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


def resolve_device(preferred: str | None = None) -> Any:
    import torch  # local import: keeps non-torch modules importable

    name = (preferred or settings.device or "auto").lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name.startswith("cuda") and not torch.cuda.is_available():
        # Fail loudly: silently falling back hides a misconfigured GPU box.
        raise RuntimeError(
            "settings.device requests CUDA but torch.cuda.is_available() is False"
        )

    return torch.device(name)
