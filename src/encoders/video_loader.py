"""
video_loader.py
Responsible for video decoding, slice by time [start, end] (seconds), sample frames
Produce raw RGB frames as PIL.Image
This is separate from encoder: encoder only does embedding.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List
import cv2
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def read_video_frames_raw(
    video_path: str | Path,
    start: float,
    end: float,
    sample_rate: float,
) -> List[Image.Image]:
    """
    Read video segment [start, end] seconds, sample at sample_rate (fps).
    Return list of PIL RGB images.
    Args:
        video_path: path to mp4
        start: start time in seconds
        end: end time in seconds
        sample_rate: target sampling fps
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning("Invalid fps for %s", video_path)
        cap.release()
        return []

    duration = (end - start)
    if duration <= 0:
        cap.release()
        return []

    sample_interval = 1.0 / sample_rate
    next_sample_ts = start

    frames: List[Image.Image] = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts < start:
            continue
        if ts > end:
            break
        if ts >= next_sample_ts:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)
            next_sample_ts += sample_interval

    cap.release()
    return frames
