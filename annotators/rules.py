"""Per-dataset annotation rules and fallback annotations.

ActivityNet annotates a simplified scene subset; NuScenes keeps the full
driving-context annotation set. ``semantic_summary`` is free text for both.
These lists are the single source for the VLM prompt, the LMStudio/Ollama output 
schema and output validation.
"""

from __future__ import annotations

from typing import Dict, List

MAX_SEMANTIC_SUMMARY_WORDS = 50

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
    "people_count": ["0", "single", "few", "crowd", "unknown"],
    "lighting": ["natural_light", "dim", "indoor_light", "unknown"],
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
    "driving_context": [
            "urban_street",
            "highway",
            "intersection",
            "roundabout",
            "tunnel",
            "bridge",
            "underpass",
            "parking_lot",
            "residential",
            "other",
            "unknown",
        ],
    "weather": ["clear", "cloudy", "rainy", "snowy", "foggy", "unknown"],
    "people_count": ["0", "single", "few", "crowd", "unknown"],
    "lighting": ["daylight", "overcast", "night", "artificial_light", "unknown"],
    "time_of_day": [
        "dawn",
        "morning",
        "noon",
        "afternoon",
        "dusk",
        "night",
        "unknown",
    ],
    "cyclist_presence": ["none", "single", "few", "many", "unknown"],
    "road_user_density": ["empty", "sparse", "moderate", "dense", "unknown"],
    "traffic_flow": ["free_flow", "moving_slow", "congested", "stopped", "unknown"],
}

DEFAULT_ANNOTATION_ACTIVITYNET: Dict[str, str] = {
    "scene_env": "unknown",
    "scene_type": "other",
    "people_count": "unknown",
    "lighting": "unknown",
    "time_of_day": "unknown",
    "semantic_summary": "unknown",
}

DEFAULT_ANNOTATION_NUSCENES: Dict[str, str] = {
    "driving_context": "unknown",
    "weather": "unknown",
    "people_count": "unknown",
    "lighting": "unknown",
    "time_of_day": "unknown",
    "cyclist_presence": "unknown",
    "road_user_density": "unknown",
    "traffic_flow": "unknown",
    "semantic_summary": "unknown",
}
