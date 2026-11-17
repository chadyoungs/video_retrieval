"""Per-dataset annotation rules and fallback annotations.

ActivityNet annotates a simplified scene subset; NuScenes keeps the full
driving-context annotation set. ``semantic_summary`` is free text for both.
These lists are the single source for the VLM prompt, the Ollama output schema
and output validation.
"""

from __future__ import annotations

from typing import Dict, List

MAX_SEMANTIC_SUMMARY_WORDS = 30

SCENE_TYPE_VALUES = [
    "street",
    "square",
    "park",
    "field",
    "mountain",
    "forest",
    "river",
    "sea",
    "gym",
    "office",
    "classroom",
    "kitchen",
    "living_room",
    "mall",
    "parking",
    "underpass",
    "other",
]

# ActivityNet: simplified scene annotation subset.
ANNOTATION_RULES_ACTIVITYNET: Dict[str, List[str]] = {
    "scene_env": ["indoor", "outdoor", "semi_outdoor", "unknown"],
    "scene_type": SCENE_TYPE_VALUES,
    "weather": ["sunny", "cloudy", "rainy", "foggy", "snowy", "unknown"],
    "lighting": ["bright", "normal", "dim", "indoor_light"],
    "time_of_day": [
        "dawn",
        "morning",
        "noon",
        "afternoon",
        "dusk",
        "night",
        "unknown",
    ],
}

# NuScenes: full driving-context annotation set.
ANNOTATION_RULES_NUSCENES: Dict[str, List[str]] = {
    **ANNOTATION_RULES_ACTIVITYNET,
    "person_count": ["0", "single", "few", "crowd"],
    "driving_context": [
        "city_driving",
        "highway",
        "intersection",
        "parking_lot",
        "residential",
        "unknown",
    ],
    "road_user_density": ["empty", "sparse", "moderate", "dense", "unknown"],
    "traffic_flow": ["free_flow", "slow", "congested", "stopped", "unknown"],
}

DEFAULT_ANNOTATION_ACTIVITYNET: Dict[str, str] = {
    "scene_env": "unknown",
    "scene_type": "other",
    "weather": "unknown",
    "lighting": "normal",
    "time_of_day": "unknown",
    "semantic_summary": "unknown",
}

DEFAULT_ANNOTATION_NUSCENES: Dict[str, str] = {
    "scene_env": "unknown",
    "scene_type": "other",
    "weather": "unknown",
    "lighting": "normal",
    "time_of_day": "unknown",
    "person_count": "0",
    "driving_context": "unknown",
    "road_user_density": "unknown",
    "traffic_flow": "unknown",
    "semantic_summary": "unknown",
}
