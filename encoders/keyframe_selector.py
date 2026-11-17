"""Keyframe selection, digested from the legacy ``utils/keyframeselection.py``.

Three frame-level heuristics vote on candidate indices:

1. **top-order** – largest frame-to-frame YUV difference;
2. **threshold** – relative change vs the previous frame exceeds ``THRESH``;
3. **local maxima** – maxima of the smoothed frame-diff signal.

Candidates are ranked by how many heuristics cite them; ties keep the lower
index. Returns at most ``top_n`` frames, or the input unchanged when it is
already short enough.

Original source: https://github.com/amanwalia123/KeyFramesExtraction
"""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence

import cv2
import numpy as np
from scipy.signal import argrelextrema

THRESH = 0.2
SMOOTH_WINDOW_SIZE = 3


def _to_yuv(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to YUV along the legacy RGB -> BGR -> YUV path."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return arr
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)


def _frame_diffs(frames: Sequence[np.ndarray]) -> List[float]:
    """Per-frame YUV difference against the previous frame; first frame is 0."""
    yuv = [_to_yuv(frame) for frame in frames]
    diffs = [0.0]
    for i in range(1, len(yuv)):
        diffs.append(float(np.sum(cv2.absdiff(yuv[i], yuv[i - 1]))))
    return diffs


def _rel_change(a: float, b: float) -> float:
    return (b - a) / max(a, b) if max(a, b) != 0 else 0.0


def _top_order_indices(values: List[float], top_n: int) -> List[int]:
    order = np.argsort(values)[::-1][:top_n]
    return [int(i) for i in order]


def _threshold_indices(values: List[float], thresh: float = THRESH) -> List[int]:
    return [
        i
        for i in range(1, len(values))
        if abs(_rel_change(values[i - 1], values[i])) >= thresh
    ]


def _smooth(
    x: np.ndarray,
    window_len: int = SMOOTH_WINDOW_SIZE,
    window: str = "hanning",
) -> np.ndarray:
    """Windowed-convolution smoothing; legacy reflection padding is kept."""
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in {"flat", "hanning", "hamming", "bartlett", "blackman"}:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', "
            "'blackman'"
        )

    padded = np.r_[
        2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]
    ]
    weights = (
        np.ones(window_len, "d")
        if window == "flat"
        else getattr(np, window)(window_len)
    )
    smoothed = np.convolve(weights / weights.sum(), padded, mode="same")
    return smoothed[window_len - 1 : -window_len + 1]


def _local_maxima_indices(
    diffs: List[float], window: int = SMOOTH_WINDOW_SIZE
) -> List[int]:
    smoothed = _smooth(np.asarray(diffs, dtype=np.float64), window)
    return [int(i) for i in argrelextrema(smoothed, np.greater)[0]]


def select_keyframes(
    frames: Sequence[np.ndarray], top_n: int
) -> List[np.ndarray]:
    """Return up to ``top_n`` keyframes; empty list for invalid input."""
    if not frames or top_n <= 0:
        return []
    if len(frames) <= top_n:
        return list(frames)

    diffs = _frame_diffs(frames)
    votes = Counter(
        _top_order_indices(diffs, top_n)
        + _threshold_indices(diffs)
        + _local_maxima_indices(diffs)
    )
    # Most votes first; deterministic tie-break on the lower frame index.
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    indices = [idx for idx, _ in ranked[:top_n]]
    return [frames[i] for i in indices]


# Legacy public name kept for backward compatibility.
select_frames = select_keyframes
