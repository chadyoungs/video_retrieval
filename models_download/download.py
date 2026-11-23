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
from pathlib import Path

# Import the HF env/config helper before ``huggingface_hub`` so HF_ENDPOINT
# is live when the hub reads its constants.
from utils.hf_models import download_repo, local_model_dir

PRESET_MAP = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
    "bge-reranker": "BAAI/bge-reranker-base",
}


def download_model(repo_id: str, force: bool = False) -> Path:
    save_dir = local_model_dir(repo_id)
    if save_dir.exists() and any(save_dir.iterdir()) and not force:
        print(f"already present: {save_dir}")
        return save_dir

    # One folder, one copy: files land directly in models/<basename>.
    download_repo(repo_id)
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
