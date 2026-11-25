from __future__ import annotations
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def segment_bounds(
    video_file: str,
    clip_duration: float = 10.0,
    clip_overlap: float = 0.0
) -> tuple[list[float], list[float]]:
    """
    Sliding window segment generator, same logic as original video_start_end_generator.
    Return (segment_starts, segment_ends).
    """
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or total_frames <= 0:
        logger.warning("Invalid video properties %s: fps=%s, frames=%s", video_file, fps, total_frames)
        return [], []

    video_duration = total_frames / fps
    if video_duration <= clip_duration:
        return [0.0], [video_duration]
    
    stride = clip_duration - clip_overlap
    if stride <= 0:
        logger.warning("Invalid clip_overlap: %s >= clip_duration: %s", clip_overlap, clip_duration)
        return [0.0], [video_duration]
        
    segment_starts = np.arange(0.0, video_duration, stride)
    segment_ends = segment_starts + clip_duration
    
    # make sure the last segment ends at the video duration
    segment_starts[-1] = max(0.0, video_duration - clip_duration)
    segment_ends[-1] = video_duration
    
    return segment_starts.tolist(), segment_ends.tolist()


def whole_video_bounds(video_file: str) -> tuple[list[float], list[float]]:
    """One segment covering the whole file, for pre-segmented datasets.

    nuScenes builder clips are already final segments; applying the
    ActivityNet sliding window (duration + overlap) to them would duplicate
    or truncate content.
    """
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or total_frames <= 0:
        logger.warning("Invalid video properties %s: fps=%s, frames=%s", video_file, fps, total_frames)
        return [], []

    return [0.0], [total_frames / fps]
