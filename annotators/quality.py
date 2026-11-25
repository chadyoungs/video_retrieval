"""Annotation quality assessment -> ``quality_flag``.

Ingestion writes one flag per segment. A segment is flagged as bad when the VLM
annotation is not trustworthy:

* an enum field carries an ``unknown`` (or otherwise unusable) value;
* ``semantic_summary`` is empty, a placeholder, mojibake, punctuation-only,
  degenerated (one repeated character/token), or a meaningless short phrase
  such as ``unknown scene`` / ``a video clip`` / ``no information``.

The reasons are returned alongside the flag so callers can log them and build
statistics. Keeping the flag policy in one place means PostgreSQL, Milvus
side-cars and retrieval filters all agree on the same definition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: Values treated as "no information" for either an enum field or the summary.
UNKNOWN_VALUES: frozenset[str] = frozenset(
    {
        "",
        "unknown",
        "unk",
        "none",
        "null",
        "nil",
        "n/a",
        "na",
        "not sure",
        "unsure",
        "unclear",
        "indeterminate",
    }
)

#: Words that carry no retrievable content on their own: articles, pronouns,
#: auxiliaries, media nouns and "I could not tell" vocabulary. A summary made
#: only of these is as useless as an empty one.
_STOPWORD_TOKENS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "this", "that", "these", "those", "it", "its", "they",
        "them", "their", "there", "here", "is", "are", "was", "were", "be",
        "been", "being", "has", "have", "had", "do", "does", "did", "can",
        "could", "would", "should", "will", "may", "might", "must", "and", "or",
        "but", "with", "without", "from", "into", "onto", "over", "under",
        "about", "across", "in", "on", "at", "of", "to", "for", "by", "as",
        "while", "during", "some", "any", "many", "much", "more", "most",
        "very", "also", "just", "only", "then", "than", "up", "down", "out",
        "video", "videos", "clip", "clips", "image", "images", "frame",
        "frames", "picture", "pictures", "photo", "photos", "footage", "shot",
        "shots", "scene", "scenes", "content", "contents", "view", "views",
        "thing", "things", "object", "objects", "person", "persons", "people",
        "someone", "something", "anything", "anyone", "everyone", "everything",
        "show", "shows", "showing", "shown", "display", "displays", "displaying",
        "appear", "appears", "appearing", "seem", "seems", "seeming", "look",
        "looks", "looking", "see", "seen", "seeing", "visible", "present",
        "possible", "possibly", "maybe", "probably", "happening", "happens",
        "going", "occurring", "shown", "unable", "impossible", "describe",
        "describes", "described", "description", "information", "info",
        "detail", "details", "nothing", "empty", "blank", "missing",
        "unspecified", "unspecific", "generic", "inconclusive", "general",
        "overall", "main", "available", "image", "photo", "frame",
    }
)

#: Uncertainty markers: if the summary still has to hedge, it is not usable.
_UNCERTAIN_TOKENS: frozenset[str] = frozenset(
    {
        "unknown", "unsure", "uncertain", "unclear", "indeterminate",
        "undetermined", "undeterminable", "indistinct", "unrecognizable",
        "indiscernible", "unidentifiable", "unknowns", "not", "no", "none",
        "cant", "cannot", "unable", "unspecified",
    }
)

#: Whole-summary Chinese fillers treated as "no information".
_CJK_MEANINGLESS: frozenset[str] = frozenset(
    {
        "未知", "无", "没有", "无内容", "空的", "空", "不清楚", "不确定",
        "无法确定", "看不出来", "无法识别", "没有内容", "没有任何内容",
        "什么都没有", "未知场景", "未知画面", "无法判断", "不详", "未识别",
        "未标注", "暂无", "暂无内容", "视频", "画面", "图片", "片段", "内容",
    }
)

#: Latin/CJK word tokens (digits included; punctuation dropped).
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

#: Punctuation acceptable in a free-text summary besides letters/digits. Noisy
#: symbols (``@#$%^&*``) count against the garbage ratio.
_ALLOWED_SYMBOLS = set(" .,;:!?'\"()[]/-_")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BRACKET_PLACEHOLDER = re.compile(r"^[\[{<][^\]}>]{0,60}[\]}>]$")

#: Ratio of unexpected symbols above which the summary is considered garbage.
_MAX_SYMBOL_RATIO = 0.3


@dataclass(frozen=True)
class QualityReport:
    """Outcome of the quality assessment for one segment."""

    flag: str
    reasons: Tuple[str, ...]

    @property
    def is_bad(self) -> bool:
        return bool(self.reasons)

    @property
    def reason_text(self) -> str:
        return ",".join(self.reasons) if self.reasons else "-"


def _is_informative_char(char: str) -> bool:
    """Letter, digit, CJK/kana/hangul or an accepted symbol."""
    code = ord(char)
    if char.isalnum():  # latin letters, digits, unicode letters (incl. CJK)
        return True
    if 0x3040 <= code <= 0x30FF:  # kana
        return True
    if 0xAC00 <= code <= 0xD7AF:  # hangul
        return True
    if 0x4E00 <= code <= 0x9FFF:  # CJK ideographs
        return True
    return char in _ALLOWED_SYMBOLS


def _word_tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _phrase_problems(text: str, strict_unknown: bool) -> Tuple[str, ...]:
    """Reject short summaries that read like "unknown scene" or "a video clip".

    Two cases: (a) the whole phrase is media/filler vocabulary, so nothing can be
    retrieved from it; (b) it still hedges with an uncertainty marker
    (``unknown``/``unclear`` ...). Both leave the segment without usable text.
    """
    tokens = _word_tokens(text)
    if not tokens:
        return ()

    # Short CJK summaries are one whitespace-less token; judge them by content.
    cjk_only = _CJK_RE.findall(text)
    if cjk_only and len(tokens) == 1:
        inner = "".join(cjk_only)
        if inner in _CJK_MEANINGLESS or len(inner) <= 2:
            return ("summary_meaningless_phrase",)
        return ()

    if all(token in _STOPWORD_TOKENS or token in _UNCERTAIN_TOKENS for token in tokens):
        return ("summary_meaningless_phrase",)

    if strict_unknown and any(token in _UNCERTAIN_TOKENS for token in tokens):
        return ("summary_uncertain",)

    # Need at least two content words: "car" or "street" alone is not a summary.
    # Numbers count (``people: 3; cars: 2`` carries real information).
    content = [
        token
        for token in tokens
        if token not in _STOPWORD_TOKENS and token not in _UNCERTAIN_TOKENS
    ]
    # 4, empirical setting in this repo
    if len(content) < 4:
        return ("summary_too_short",)
    return ()


def _summary_problems(summary: Any, strict_unknown: bool = True) -> Tuple[str, ...]:
    """Return the reasons why ``semantic_summary`` is not usable."""
    if summary is None:
        return ("summary_missing",)

    text = str(summary).strip()
    if not text:
        return ("summary_empty",)
    if text.lower() in UNKNOWN_VALUES:
        return ("summary_unknown",)
    if "\ufffd" in text or _CONTROL_CHARS.search(text):
        return ("summary_mojibake",)
    if _BRACKET_PLACEHOLDER.match(text):
        return ("summary_placeholder",)
    if len(text) > 512:
        return ("summary_too_long",)

    informative = [char for char in text if _is_informative_char(char)]
    if not any(char.isalnum() for char in informative):
        return ("summary_no_text",)

    symbol_ratio = 1.0 - (len(informative) / len(text))
    if symbol_ratio > _MAX_SYMBOL_RATIO:
        return ("summary_garbage_chars",)

    letters = {char.lower() for char in text if char.isalnum()}
    if len(text) >= 4 and len(letters) <= 1:
        return ("summary_degenerate_chars",)

    tokens = [token for token in re.split(r"\s+", text) if token]
    if len(tokens) >= 3 and len({token.lower() for token in tokens}) == 1:
        return ("summary_repeated_tokens",)

    return _phrase_problems(text, strict_unknown)


def assess_annotation_quality(
    annotation: Optional[Mapping[str, Any]],
    rules: Mapping[str, Sequence[str]],
    *,
    ok_flag: str = "ok",
    bad_flag: str = "bad",
    strict_unknown: bool = True,
    ignore_unknown_fields: Sequence[str] = (),
) -> QualityReport:
    """Assess one annotation dict and return its quality flag.

    Args:
        annotation: annotation dict produced by the annotator.
        rules: enum field -> allowed values (``BaseAnnotator.rules``).
        ok_flag: flag value for a usable annotation.
        bad_flag: flag value for an unusable annotation.
        strict_unknown: flag any ``unknown`` enum value as bad.
        ignore_unknown_fields: enum fields where ``unknown`` is tolerated.
    """
    if ok_flag == bad_flag:
        raise ValueError("ok_flag and bad_flag must differ")

    annotation = annotation or {}
    ignored = {field.lower() for field in ignore_unknown_fields}
    reasons: list[str] = []

    for field, allowed in rules.items():
        value = annotation.get(field)
        normalized = "" if value is None else str(value).strip().lower()
        if normalized in UNKNOWN_VALUES:
            if strict_unknown and field.lower() not in ignored:
                reasons.append(f"unknown:{field}")
            continue
        if allowed and normalized not in {str(item).lower() for item in allowed}:
            reasons.append(f"invalid:{field}")

    reasons.extend(_summary_problems(annotation.get("semantic_summary"), strict_unknown))

    flag = bad_flag if reasons else ok_flag
    return QualityReport(flag=flag, reasons=tuple(reasons))
