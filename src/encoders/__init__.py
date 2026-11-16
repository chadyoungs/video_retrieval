"""
Encoders & VLM Annotation Module
Two representation paths:
1. KeyFrame based: extract keyframe -> VLM annotate -> image embedding
2. Video sequence encoder (CLIP4CLIP): frame sequence -> temporal aggregation -> video embedding
Annotation output is structured metadata for rule-based retrieval / rule reranker
"""
