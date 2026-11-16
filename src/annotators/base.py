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

from app.config.settings import settings
from app.annotators.rules import MAX_SEMANTIC_SUMMARY_WORDS

logger = logging.getLogger(__name__)


def _image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base-64 JPEG string."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class BaseAnnotator(abc.ABC):
    """Abstract annotator interface shared by dataset implementations."""

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
    def dataset_key(self) -> str:
        """Return the dataset key this annotator serves."""

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def annotate(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Annotate keyframes; return a validated annotation dict.

        Falls back to the dataset default annotation when no images are
        supplied, the VLM call fails after retries, or the output fails
        validation.
        """
        if not images:
            logger.warning("%s annotator received no images", self.dataset_key())
            return dict(self.default_annotation)

        # Keep at most the configured number of VLM frames.
        frames = images[: self.n_vlm_frames]
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
        """Strict annotation prompt, digesting the legacy prompt wording."""
        rules_json = json.dumps(self.rules, indent=2)
        field_lines = "\n".join(
            f'            "{field}": "[value only from list]"'
            for field in self.rules
        )
        return (
            "You are a strict video scene annotation expert.\n"
            "YOU MUST FOLLOW ALL RULES 100% STRICTLY.\n"
            "YOU ONLY OUTPUT A SINGLE JSON OBJECT, NO OTHER TEXT, "
            "NO EXPLANATION, NO MARKDOWN, NO CHATTING.\n\n"
            "### RULES (YOU CANNOT CHOOSE VALUES OUTSIDE THESE LISTS)\n"
            f"{rules_json}\n\n"
            "### EXTRA RULES\n"
            "- semantic_summary must be a concise summary "
            f"(max {MAX_SEMANTIC_SUMMARY_WORDS} words).\n\n"
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
                        self.dataset_key(),
                        attempt + 1,
                    )
                    continue

                return json.loads(content)

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "%s annotator network error (attempt %d): %s",
                    self.dataset_key(),
                    attempt + 1,
                    exc,
                )
            except json.JSONDecodeError as exc:
                logger.warning(
                    "%s annotator JSON decode error (attempt %d): %s",
                    self.dataset_key(),
                    attempt + 1,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "%s annotator unexpected error (attempt %d): %s",
                    self.dataset_key(),
                    attempt + 1,
                    exc,
                )

        return None

    # ------------------------------------------------------------------ #
    # Output validation (legacy validate_annotation_output口径)
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
