"""Hugging Face model location + loading helpers.

All encoder / reranker weights live under ``settings.model_cache_dir`` in a
folder named after the repo basename (``openai/clip-vit-base-patch32`` ->
``models/clip-vit-base-patch32``). A relative cache dir is resolved against the
repository root so the pipeline behaves the same whatever the current working
directory is.

The endpoint must be applied **before** ``transformers`` / ``huggingface_hub``
is imported: the hub caches ``HF_ENDPOINT`` in its constants at import time, so
setting the env var inside ``from_pretrained`` is too late. This module applies
it at import time and refreshes the cached constant as a fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from config.settings import settings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def cache_root() -> Path:
    root = Path(settings.model_cache_dir).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def local_model_dir(model_name: str) -> Path:
    """Folder this repo keeps for ``model_name``."""
    return cache_root() / model_name.split("/")[-1]


def _apply_hf_env() -> None:
    """Export the configured endpoint and refresh the hub's cached constant."""
    if not settings.hf_endpoint:
        return
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint
    try:
        import huggingface_hub.constants as hf_const

        hf_const.ENDPOINT = settings.hf_endpoint
    except ImportError:
        pass


# Apply as early as possible: encoder modules must import this module before
# ``transformers`` so the endpoint is live when the hub reads it.
_apply_hf_env()


def _hub_kwargs() -> Dict[str, Any]:
    """Hub-side download kwargs: land weights in our own cache root."""
    return {"cache_dir": str(cache_root())}


def load_model(model_name: str, model_cls: Any) -> Any:
    """Load ``model_cls`` for ``model_name`` from the local cache or hub."""
    folder = local_model_dir(model_name)
    if folder.exists() and any(folder.iterdir()):
        logger.info("Loading %s from local cache %s", model_name, folder)
        return model_cls.from_pretrained(folder, local_files_only=True)

    if settings.hf_local_only:
        raise FileNotFoundError(
            f"Model {model_name} is not present under {folder} and "
            f"settings.hf_local_only=True. Run: python -m models_download.download "
            f"--repo-id {model_name}"
        )

    logger.warning(
        "Model %s not found under %s; downloading from %s",
        model_name,
        folder,
        settings.hf_endpoint or "the default hub",
    )
    model = model_cls.from_pretrained(model_name, **_hub_kwargs())
    folder.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(folder)
    except Exception:  # noqa: BLE001 - cache write is best effort
        logger.warning("Could not persist %s into %s", model_name, folder, exc_info=True)
    return model


def load_processor(model_name: str, processor_cls: Any) -> Any:
    """Load the matching processor / tokenizer for ``model_name``."""
    folder = local_model_dir(model_name)
    if folder.exists() and any(folder.iterdir()):
        return processor_cls.from_pretrained(folder, local_files_only=True)
    return processor_cls.from_pretrained(model_name, **_hub_kwargs())
