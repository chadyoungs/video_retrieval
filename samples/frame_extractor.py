"""
Fast frame extractor for video, seek to target timestamp instead of sequential scan.
Support: extract single frame at given second, save to png/jpg
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2


def extract_frame_at_second(
    video_path: str | Path,
    time_sec: float,
    dest_path: str | Path,
) -> Path | None:
    video_path = Path(video_path)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"⚠️ Invalid fps={fps}, fallback to sequential read")
            return _extract_sequential(cap, time_sec, dest_path)

        target_frame = int(round(time_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(str(dest_path), frame)
            print(f"✅ Frame saved: {dest_path} | target sec={time_sec:.2f}, frame={target_frame}")
            return dest_path
        else:
            print(f"⚠️ Seek failed, fall back sequential scan")
            return _extract_sequential(cap, time_sec, dest_path)
    finally:
        cap.release()


def _extract_sequential(cap: cv2.VideoCapture, time_sec: float, dest_path: Path) -> Path | None:
    """Fallback: read frame by frame, compatible for videos with bad seek support"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0
    target_frame = int(round(time_sec * fps))
    current_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Reached EOF before target frame {target_frame}")
            return None
        if current_idx == target_frame:
            cv2.imwrite(str(dest_path), frame)
            print(f"✅ Frame saved (sequential): {dest_path}")
            return dest_path
        current_idx += 1


def main():
    parser = argparse.ArgumentParser(description="Extract one frame from video at specified time(second)")
    parser.add_argument("--video", type=str, required=True, help="input video path")
    parser.add_argument("--time-sec", type=float, required=True, help="target timestamp in seconds")
    parser.add_argument("--dest", type=str, required=True, help="output image path (png/jpg)")
    args = parser.parse_args()

    result = extract_frame_at_second(args.video, args.time_sec, args.dest)
    exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()
