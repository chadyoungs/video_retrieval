from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    
    # ========= Milvus =========
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_alias: str = "activitynet-demo"
    # dataset -> milvus collection mapping
    milvus_collections: dict[Literal["activitynet", "nuscenes"], str] = {
        "activitynet": "activitynet_clip_embedding",
        "nuscenes": "nuscenes_clip_embedding",
    }
    # legacy single collection (old code fallback)
    collection_name: str = "video_reverse_search_demo_index"
    milvus_metric_type: str = "COSINE"
    
    # ======== PostgreSQL =========
    pg_dsn: str = "postgresql://user:password@127.0.0.1:5432/video_retrieval"
    
    # ========= Ollama VLM Annotation =========
    annotation_enable: bool = True
    n_vlm_frames: int = 4    # VLM annotation frames per segment
     
    ollama_api_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "qwen3-vl:8b"
    ollama_timeout: int = 600
    max_retries: int = 3
    retry_backoff_factor: float = 1.5

    # ========= Semantic Encoder / CLIP / SigLIP =========
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

    @property
    def annotation_model_version(self) -> str:
        return self.ollama_model

    # ========= Dataset Source =========
    dataset_source: Literal["activitynet", "nuscenes"] = "activitynet"

    # ========= ActivityNet Dataset paths =========
    activitynet_data_root: str = "/mnt/sdc/activitynet_caption/demo_data"
    activitynet_file_format: str = "mp4"

    # ========= NuScenes Dataset paths =========
    nuscenes_video_root: str = "/mnt/sdb/datasets/nuscenes-data/nuscenes/samples/CAM_FRONT"
    nuscenes_clip_mapping_file: str = "./datasets/builders/clip_mapping.json"
    nuscenes_dataset_name: str = "nuscenes"
    nuscenes_dataset_version: str = "v1.0-mini"
    nuscenes_dataset_split: str = "train"
    nuscenes_camera_channel: str = "CAM_FRONT"
    
    nuscenes_meta_root: str = "/mnt/sdb/datasets/nuscenes-data/nuscenes/{nuscenes_dataset_version}"
    builder_output_fps: float = 12.0
    
    # ========= Model Cache =========
    model_cache_dir: str = "./models"
    hf_endpoint: str = "https://hf-mirror.com"
    
    # ========= Frame / Embedding Policy =========
    frame_policy: Literal["whole_clip", "keyframes"] = "whole_clip"
    keyframe_count: int = 4  # valid while frame_policy="keyframes"
   
    # ========= Clip slicing =========
    clip_duration_sec: float = 10.0
    frame_sampling_rate: int = 10
    clip_overlap_sec: float = 0.0

    # ========= Batch / Compute =========
    batch_video: int = 2
    batch_size_db: int = 5
    num_workers: int = 1
    num_processes: int = 2

    # ========= Rerank =========
    reranker_top_k: int = 10
    reranker_score_threshold: float = 0.0
    reranker_weight_rule: float = 0.4
    reranker_weight_cross_encoder: float = 0.3
    reranker_weight_vlm: float = 0.3
    cross_encoder_model_name: str = "BAAI/bge-reranker-base"


settings = Settings()
