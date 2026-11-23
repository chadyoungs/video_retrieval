"""Single source of runtime configuration, overridable via environment / ``.env``.

Every attribute here is read with ``settings.<name>``; the matching environment
variable is the upper-cased name (e.g. ``PG_DSN``, ``DATASET_SOURCE``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ======== Device =========
    # "auto" picks cuda when torch reports it, otherwise cpu. Use "cpu" on
    # macOS / CPU-only boxes. "cuda" forces CUDA and fails loudly if missing.
    device: str = "auto"

    # ========= Milvus =========
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_alias: str = "video_retrieval"
    # dataset -> milvus collection mapping
    milvus_collections: dict[Literal["activitynet", "nuscenes"], str] = {
        "activitynet": "activitynet_clip_embedding",
        "nuscenes": "nuscenes_clip_embedding",
    }
    # legacy single collection
    collection_name: str = "video_reverse_search_demo_index"
    milvus_metric_type: str = "COSINE"

    # ======== PostgreSQL =========
    pg_dsn: str = "postgresql://devuser:pass123@127.0.0.1:5432/postgres"

    # ========= Ollama VLM Annotation =========
    annotation_enable: bool = True
    n_vlm_frames: int = 4  # VLM annotation frames per segment

    ollama_host: str = "http://localhost:11434"
    ollama_api_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "qwen3-vl:8b"
    ollama_timeout: int = 600
    max_retries: int = 3
    retry_backoff_factor: float = 1.5

    @property
    def annotation_model_version(self) -> str:
        return self.ollama_model

    # ========= Semantic Encoder / CLIP / SigLIP =========
    # ========= huggingface Model Cache =========
    model_cache_dir: str = "./models"
    hf_endpoint: str = "https://hf-mirror.com"
    # True: only load weights already present under model_cache_dir, never
    # touch the network. False: download into model_cache_dir on first use.
    hf_local_only: bool = False

    # ========= Frame / Embedding Policy =========
    encoder_type: Literal["whole_clip", "keyframes"] = "keyframes"
    keyframe_count: int = 5  # valid while encoder_type="keyframes"

    semantic_model_tier: Literal["base", "advanced"] = "base"
    base_clip_model_name: str = "openai/clip-vit-base-patch32"
    advanced_clip_model_name: str = "google/siglip-so400m-patch14-384"
    embedding_dim_base: int = 512
    embedding_dim_advanced: int = 1152

    @property
    def clip_model_name(self) -> str:
        if self.semantic_model_tier == "advanced":
            return self.advanced_clip_model_name
        return self.base_clip_model_name

    @property
    def embedding_dim(self) -> int:
        if self.semantic_model_tier == "advanced":
            return self.embedding_dim_advanced
        return self.embedding_dim_base

    @property
    def embedding_model_version(self) -> str:
        return self.clip_model_name

    # ========= Dataset Source =========
    dataset_source: Literal["activitynet", "nuscenes"] = "activitynet"

    # ========= ActivityNet Dataset paths =========
    activitynet_data_root: str = "/mnt/sdc/activitynet_caption/demo_data"
    activitynet_file_format: str = "mp4"

    # ========= NuScenes Dataset paths =========
    nuscenes_raw_dataroot: str = "/mnt/sdb/datasets/nuscenes-data/nuscenes"
    nuscenes_video_root: str = "/mnt/sdb/datasets/nuscenes-data/nuscenes/samples/CAM_FRONT"
    nuscenes_clip_mapping_file: str = "./datasets/builders/clip_mapping.json"
    nuscenes_dataset_name: str = "nuscenes"
    nuscenes_dataset_version: str = "v1.0-mini"
    nuscenes_dataset_split: str = "train"
    nuscenes_camera_channel: str = "CAM_FRONT"

    nuscenes_meta_root: str = "/mnt/sdb/datasets/nuscenes-data/nuscenes/v1.0-mini"
    builder_output_fps: float = 2.0

    # ========= Clip slicing =========
    clip_duration_sec: float = 10.0
    frame_sampling_rate: int = 10
    clip_overlap_sec: float = 2.0
    min_clip_sec: float = 2.0

    # ========= Batch / Compute =========
    batch_video: int = 1
    batch_size_db: int = 5  # valid for both pg and milvus
    num_workers: int = 1
    num_processes: int = 1

    # ========= Hybrid recall / Rerank =========
    weight_vector: float = 0.7
    weight_metadata: float = 0.3
    recall_pool_multiplier: int = 2

    reranker_top_k: int = 10
    reranker_score_threshold: float = 0.0
    reranker_weight_rule: float = 0.4
    reranker_weight_cross_encoder: float = 0.3
    reranker_weight_vlm: float = 0.3
    # Cross-encoder / VLM are optional: when a component is disabled, has no
    # weights or its endpoint is unreachable, the remaining signals are
    # renormalized instead of being padded with the coarse score.
    reranker_enable_cross_encoder: bool = True
    reranker_enable_vlm: bool = True
    ollama_rerank_model: str = "qwen3-vl:8b"
    ollama_rerank_timeout: int = 30
    cross_encoder_model_name: str = "BAAI/bge-reranker-base"
    ce_max_seq_len: int = 512
    ce_batch_size: int = 16

    #: Alias kept for older call sites / scripts.
    @property
    def reranker_candidates(self) -> int:
        return self.reranker_top_k

    @property
    def ollama_generate_url(self) -> str:
        return f"{self.ollama_host.rstrip('/')}/api/generate"

    def rerank_conf(self, dataset_name: str) -> dict[str, float]:
        """Per-dataset rerank weights. Both datasets share the same policy today."""
        if dataset_name not in ("activitynet", "nuscenes"):
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        return {
            "rule_weight": self.reranker_weight_rule,
            "ce_weight": self.reranker_weight_cross_encoder,
            "vlm_weight": self.reranker_weight_vlm,
        }


settings = Settings()
