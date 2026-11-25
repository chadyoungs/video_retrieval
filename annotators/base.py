"""Shared VLM annotator base.

The base owns the Ollama call, retry / backoff, prompt and output schema
construction, and legacy-style validation. Subclasses only declare their rule
set, fallback annotation and dataset key.

The annotator does not read video files; keyframes are supplied by the
pipeline as PIL images.
"""

from __future__ import annotations

import abc
import base64
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from config.settings import settings
from annotators.rules import MAX_SEMANTIC_SUMMARY_WORDS

logger = logging.getLogger(__name__)


def _image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base-64 JPEG string."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class BaseAnnotator(abc.ABC):
    """Abstract annotator interface shared by dataset implementations.
    Annotate one continuous video segment (multiple input frames) and return ONE unified annotation dict for this clip.
    Note: this module does not evaluate annotation quality, even though the dedicated quality module is placed in the 
    same subfolder.
    """

    rules: Dict[str, List[str]] = {}
    default_annotation: Dict[str, str] = {}

    def __init__(self) -> None:
        self.api_url = settings.ollama_api_url
        self.model = settings.ollama_model
        self.timeout = settings.ollama_timeout
        self.max_retries = settings.max_retries
        self.backoff = settings.retry_backoff_factor
        self.n_vlm_frames = settings.n_vlm_frames

    @abc.abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset name this annotator serves."""

    def annotate(
        self,
        video_path: str,
        start: float,
        end: float,
        sample_rate: float,
        preloaded_frames: Optional[List[Image.Image]] = None
    ) -> Dict[str, Any]:
        """Annotate keyframes of one video segment; return a validated annotation dict.

        These frames are sampled from the same continuous video clip.
        Synthesize ONE unified annotation for this whole segment, not per-frame result.

        Falls back to the dataset default annotation when no images are
        supplied, the VLM call fails after retries, or the output fails
        validation.

        Args:
            video_path: path of source video (for logging only)
            start: segment start timestamp, seconds
            end: segment end timestamp, seconds
            sample_rate: frame sample rate, reserved for fallback sampling
            preloaded_frames: pre-extracted PIL frames for this clip, priority used
        """
        if not preloaded_frames:
            logger.warning("%s annotator received no images for segment [%.2f ~ %.2f] file=%s",
                           self.dataset_name(), start, end, video_path)
            return dict(self.default_annotation)

        # Keep at most the configured number of VLM frames.
        frames = preloaded_frames[: self.n_vlm_frames]
        images_b64 = [_image_to_base64(image) for image in frames]
        prompt = self._build_prompt()
        output = self._call_vlm(images_b64, prompt)
        if output is None:
            return dict(self.default_annotation)

        normalized = self._validate(output)
        return normalized if normalized else dict(self.default_annotation)

    # ------------------------------------------------------------------ #
    # VLM call
    # ------------------------------------------------------------------ #
    def _format_schema(self) -> Dict[str, Any]:
        """Ollama ``format`` schema derived from this annotator's rules."""
        return {
            "type": "object",
            "properties": {
                **{
                    field: {"type": "string", "enum": values}
                    for field, values in self.rules.items()
                },
                "semantic_summary": {"type": "string"},
            },
            "required": list(self.rules.keys()) + ["semantic_summary"],
        }

    def _build_prompt(self) -> str:
        """Strict annotation prompt.
        Important: tell VLM these frames belong to one continuous video segment, output ONE unified result.
        """
        rules_json = json.dumps(self.rules, indent=2)
        field_lines = "\n".join(
            f'            "{field}": "[value only from list]"'
            for field in self.rules
        )
        return (
            "You are a strict video scene annotation expert.\n"
            "THESE IMAGES ARE KEYFRAMES FROM THE SAME CONTINUOUS VIDEO SEGMENT. "
            "LOOK AT ALL FRAMES TOGETHER AND PRODUCE ONLY ONE UNIFIED ANNOTATION FOR THIS WHOLE CLIP. "
            "DO NOT produce separate annotation for each image.\n\n"
            "YOU MUST FOLLOW ALL RULES 100% STRICTLY.\n"
            "YOU ONLY OUTPUT A SINGLE JSON OBJECT, NO OTHER TEXT, "
            "NO EXPLANATION, NO MARKDOWN, NO CHATTING.\n\n"
            "### RULES (YOU CANNOT CHOOSE VALUES OUTSIDE THESE LISTS)\n"
            f"{rules_json}\n\n"
            "### EXTRA RULES\n"
            f"- semantic_summary must be a concise summary (max {MAX_SEMANTIC_SUMMARY_WORDS} words).\n\n"
            "### OUTPUT FORMAT (ONLY JSON, NO OTHER CONTENT)\n"
            "{\n"
            f"{field_lines}\n"
            '            "semantic_summary": "[short plain-text summary]"\n'
            "        }\n\n"
            "NOW ANNOTATE THE PROVIDED IMAGES.\n"
            "ONLY OUTPUT JSON."
        )

    def _call_vlm(
        self, images_b64: List[str], prompt: str
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt, "images": images_b64}
            ],
            "stream": False,
            "format": self._format_schema(),
            "options": {"think": False},
        }

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    time.sleep(self.backoff ** (attempt - 1))

                resp = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()

                content = (
                    resp.json().get("message", {}).get("content", "").strip()
                )
                if not content:
                    logger.warning(
                        "%s annotator: empty VLM response (attempt %d)",
                        self.dataset_name(),
                        attempt + 1,
                    )
                    continue

                return json.loads(content)

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "%s annotator network error (attempt %d): %s",
                    self.dataset_name(),
                    attempt + 1,
                    exc,
                )
            except json.JSONDecodeError as exc:
                logger.warning(
                    "%s annotator JSON decode error (attempt %d): %s",
                    self.dataset_name(),
                    attempt + 1,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "%s annotator unexpected error (attempt %d): %s",
                    self.dataset_name(),
                    attempt + 1,
                    exc,
                )

        return None

    # ------------------------------------------------------------------ #
    # Output validation (legacy validate_annotation_output)
    # ------------------------------------------------------------------ #
    def _validate(self, output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(output, dict):
            return None
        
        for field in self.rules:
            if field not in output:
                return None
        
        for field, allowed in self.rules.items():
            field_value = str(output[field]).strip().lower()
            if field_value not in allowed:
                return None
        
        summary = str(output.get("semantic_summary", "")).strip()
        if not summary or len(summary.split()) > MAX_SEMANTIC_SUMMARY_WORDS:
            return None
        
        normalized: Dict[str, Any] = {
            field: str(output[field]).strip().lower()
            for field in self.rules
        }
        normalized["semantic_summary"] = summary
        
        return normalized
