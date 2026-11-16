"""
Model weight downloader.
Support preset: clip / siglip
Read config from settings, skip if local folder already exists, support --force
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from transformers import AutoModel, AutoProcessor

from src.config.settings import settings

# preset map: preset name -> hf repo id
PRESET_MAP = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
}


def download_model(preset: str, force: bool = False) -> Path:
    if preset not in PRESET_MAP:
        raise ValueError(f"preset [{preset}] not supported, available: {list(PRESET_MAP.keys())}")

    repo_id = PRESET_MAP[preset]
    cache_root = Path(settings.model_cache_dir)
    save_dir = cache_root / preset
    cache_root.mkdir(parents=True, exist_ok=True)

    # set hf mirror
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    if save_dir.exists() and any(save_dir.iterdir()) and not force:
        print(f"✅ Model {preset} already exists at {save_dir}, skip download.")
        return save_dir

    print(f"🔽 Start downloading {preset} from {repo_id} ...")
    model = AutoModel.from_pretrained(repo_id)
    processor = AutoProcessor.from_pretrained(repo_id)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"✅ Saved to {save_dir}")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--preset",
        type=str,
        default="clip",
        choices=list(PRESET_MAP.keys()),
        help="model preset, default clip"
    )
    parser.add_argument("--force", action="store_true", help="force re-download even if exists")
    args = parser.parse_args()
    download_model(args.preset, args.force)


if __name__ == "__main__":
    main()
