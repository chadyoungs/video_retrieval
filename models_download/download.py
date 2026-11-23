"""Model weight downloader kept in sync with ``utils.hf_models``.

Weights are stored as ``<model_cache_dir>/<repo basename>`` so encoders and
rerankers find them by the same path. Local-only mode still works offline.

Examples
--------
    python -m models_download.download --preset clip
    python -m models_download.download --repo-id BAAI/bge-reranker-base
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from transformers import AutoModel, AutoProcessor

from config.settings import settings
from utils.hf_models import cache_root, local_model_dir

PRESET_MAP = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
    "bge-reranker": "BAAI/bge-reranker-base",
}


def download_model(repo_id: str, force: bool = False) -> Path:
    save_dir = local_model_dir(repo_id)
    cache_root().mkdir(parents=True, exist_ok=True)

    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    if save_dir.exists() and any(save_dir.iterdir()) and not force:
        print(f"already present: {save_dir}")
        return save_dir

    print(f"downloading {repo_id} -> {save_dir}")
    model = AutoModel.from_pretrained(repo_id)
    model.save_pretrained(save_dir)
    try:
        processor = AutoProcessor.from_pretrained(repo_id)
        processor.save_pretrained(save_dir)
    except Exception:  # noqa: BLE001 - rerankers only need a tokenizer
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(repo_id).save_pretrained(save_dir)
    print(f"saved: {save_dir}")
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights into the local cache")
    parser.add_argument("--preset", choices=sorted(PRESET_MAP), default="clip")
    parser.add_argument("--repo-id", default=None, help="Any HF repo id; overrides --preset")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    repo_id = args.repo_id or PRESET_MAP[args.preset]
    download_model(repo_id, args.force)


if __name__ == "__main__":
    main()
