"""Visual inspection of retrieval results.

Renders one row per hit: a few frames sampled from the segment plus a metrics
panel (scores, quality flag, annotation summary). The PNG pops up with the
system image viewer / browser (stdlib ``webbrowser``) - no GUI toolkit to
install. Headless servers keep the PNG and log its path.

Usage (CLI)::

    python -m retrieval.main --type text --query "playing guitar" --show

Usage (API)::

    from retrieval.viewer import show_results
    show_results(results, query="playing guitar")
"""

from __future__ import annotations

import logging
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_DIR = REPO_ROOT / "retrieval_views"


def _setup_matplotlib():
    """Render with Agg only - no GUI toolkit required.

    The popup is the system image viewer / browser (stdlib ``webbrowser``),
    so nothing beyond matplotlib itself needs to be installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Best-effort CJK font so Chinese summaries do not render as boxes.
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in (
        "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei",
    ):
        if name in available:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    return matplotlib, plt


def _segment_frames(row: Dict[str, Any], count: int) -> List[Any]:
    """Evenly spaced frames from the hit's [start, end) window."""
    from encoders.video_loader import read_video_frames_raw

    path = row.get("video_full_path") or ""
    start = float(row.get("segment_start") or 0.0)
    end = float(row.get("segment_end") or start + 1.0)
    if not path or end <= start:
        return []
    try:
        frames = read_video_frames_raw(path, start, end, sample_rate=1.0)
    except Exception:  # noqa: BLE001 - viewer must never break retrieval
        logger.warning("Could not decode frames for %s", path, exc_info=True)
        return []
    if not frames:
        return []
    if len(frames) <= count:
        return frames
    step = (len(frames) - 1) / (count - 1)
    return [frames[round(i * step)] for i in range(count)]


def _metrics_text(row: Dict[str, Any]) -> str:
    lines = [
        f"#{row.get('rank')}  {row.get('segment_id')}",
        f"[{row.get('segment_start')}s - {row.get('segment_end')}s]  "
        f"quality: {row.get('quality_flag')}",
        f"vec {row.get('vec_score', 0):.3f}   sql {row.get('sql_score', 0):.3f}   "
        f"coarse {row.get('coarse_score', 0):.3f}",
        f"rerank {row.get('rerank_score', 0):.3f}   "
        f"rule {_fmt(row.get('score_rule'))}  ce {_fmt(row.get('score_cross_encoder'))}  "
        f"vlm {_fmt(row.get('score_vlm'))}",
    ]
    annotation = row.get("annotation") or {}
    summary = annotation.get("semantic_summary")
    if summary:
        lines.append(textwrap.fill(str(summary), width=46))
    tags = {k: v for k, v in annotation.items() if k != "semantic_summary" and v}
    if tags:
        lines.append(textwrap.fill(
            "  ".join(f"{k}={v}" for k, v in tags.items()), width=46
        ))
    if row.get("rerank_reason"):
        lines.append(textwrap.fill(f"reason: {row['rerank_reason']}", width=46))
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def show_results(
    results: List[Dict[str, Any]],
    query: str = "",
    frames_per_segment: int = 3,
    save_dir: Optional[str] = None,
) -> Optional[Path]:
    """Render the hits to a PNG and pop it up with the system viewer.

    Headless servers get the PNG path in the log. Returns the saved path, or
    None when there is nothing to show.
    """
    if not results:
        logger.info("No results to display")
        return None

    _, plt = _setup_matplotlib()
    import numpy as np

    n = len(results)
    cols = frames_per_segment + 2  # frames + wide metrics panel (2 col widths)
    fig = plt.figure(figsize=(4.2 * cols, 2.6 * n + 0.8))
    grid = fig.add_gridspec(
        n, cols, width_ratios=[1] * frames_per_segment + [2, 2], hspace=0.25, wspace=0.08
    )
    fig.suptitle(
        f"Retrieval results  |  query: {query!r}  |  {n} hits", fontsize=13
    )

    for i, row in enumerate(results):
        frames = _segment_frames(row, frames_per_segment)
        for j in range(frames_per_segment):
            ax = fig.add_subplot(grid[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(frames):
                ax.imshow(np.asarray(frames[j]))
                duration = float(row.get("segment_end") or 0) - float(
                    row.get("segment_start") or 0
                )
                ts = float(row.get("segment_start") or 0) + duration * (
                    j / max(frames_per_segment - 1, 1)
                )
                ax.set_title(f"~{ts:.1f}s", fontsize=8)
            else:
                ax.text(
                    0.5, 0.5, "no frame\n(video missing?)",
                    ha="center", va="center", fontsize=9, color="gray",
                    transform=ax.transAxes,
                )
                ax.set_facecolor("#eeeeee")

        ax_text = fig.add_subplot(grid[i, frames_per_segment:])
        ax_text.axis("off")
        ax_text.text(
            0.01, 0.95, _metrics_text(row),
            va="top", ha="left", fontsize=9, family="monospace",
            transform=ax_text.transAxes,
        )

    out_dir = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"retrieval_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Result view saved to %s", save_path)

    # Pop up the PNG with the system image viewer / browser (stdlib, zero
    # extra dependencies). A no-op on headless servers: the log carries the
    # path instead.
    import webbrowser

    try:
        opened = webbrowser.open(save_path.resolve().as_uri())
    except Exception:  # noqa: BLE001 - viewer must never break retrieval
        opened = False
    if opened:
        logger.info("Opened result view in the system viewer: %s", save_path)
    else:
        logger.warning("No desktop viewer available. View the PNG: %s", save_path)
    return save_path
