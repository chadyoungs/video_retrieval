"""Video decoding: slice ``[start, end)`` seconds and sample at ``sample_rate`` fps.

Returns RGB ``PIL.Image`` frames. Frame timestamps are derived from the decoded
frame index, not ``CAP_PROP_POS_MSEC`` (that property returns 0 for some
codecs, which used to stall the sampling loop).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

logger = logging.getLogger(__name__)


def read_video_frames_raw(
    video_path: str | Path,
    start: float,
    end: float,
    sample_rate: float,
) -> List["Image.Image"]:
    """Decode ``[start, end)`` and sample at ``sample_rate`` fps.

    Args:
        video_path: mp4 / frame container path.
        start: segment start in seconds.
        end: segment end in seconds.
        sample_rate: target sampling fps (> 0).

    Returns:
        List of RGB PIL images; empty when the file cannot be decoded.
    """
    import cv2
    from PIL import Image

    if sample_rate <= 0:
        logger.error("Invalid sample_rate=%s for %s", sample_rate, video_path)
        return []

    video_path = Path(video_path)
    if not video_path.exists():
        logger.error("Video file missing: %s", video_path)
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video %s", video_path)
        return []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning("Invalid fps for %s", video_path)
            return []

        if end <= start:
            return []

        start_frame = max(0, int(round(start * fps)))
        # Seek instead of decoding from frame 0: ingestion slices many short
        # segments out of one long video.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx != start_frame:
            logger.debug(
                "Seek to frame %d landed on %d (%s)", start_frame, frame_idx, video_path
            )

        interval = 1.0 / sample_rate
        next_sample_ts = start
        frames: List["Image.Image"] = []

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            ts = frame_idx / fps
            frame_idx += 1

            if ts < start:
                continue
            if ts > end:
                break
            if ts >= next_sample_ts:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                next_sample_ts += interval

        return frames
    finally:
        cap.release()
