#!/usr/bin/env python
"""
Generate a keypoint color legend PNG that matches JABS' KEYPOINT_COLOR_MAP.

This script is used to generate a new keypoint legend image (e.g., for documentation)
if the keypoint set or colors change.

Usage:
  python dev/generate_keypoint_legend.py --out keypoint-legend.png --cols 4 --title "JABS Keypoints"
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Import the actual map so colors 100% match what's used in the app
from jabs.pose_estimation import PoseEstimation  # provides KeypointIndex
from jabs.ui.colors import KEYPOINT_COLOR_MAP  # your QColor map or RGB map


def qcolor_to_rgb_tuple(qc: object) -> tuple[float, float, float]:
    """Return matplotlib RGB tuple in [0,1] from either QColor or 0-255 triplet."""
    try:
        # If it's a Qt QColor
        r, g, b, _ = qc.getRgb()
        return (r / 255.0, g / 255.0, b / 255.0)
    except AttributeError:
        # Assume (r,g,b) in 0-255 or 0-1
        r, g, b = qc
        if max(r, g, b) > 1.0:
            return (r / 255.0, g / 255.0, b / 255.0)
        return (r, g, b)


def collect_entries() -> list[tuple[str, tuple[float, float, float]]]:
    """Collect (label, rgb) entries from KEYPOINT_COLOR_MAP in order of PoseEstimation.KeypointIndex."""
    entries = []
    # Preserve the order as defined in PoseEstimation.KeypointIndex
    for kp in PoseEstimation.KeypointIndex:
        raw_name = getattr(kp, "name", str(kp))
        parts = raw_name.split("_")
        name = " ".join(word.capitalize() for word in parts)
        color = KEYPOINT_COLOR_MAP[kp]
        rgb = qcolor_to_rgb_tuple(color)
        entries.append((name, rgb))
    return entries


def draw_legend(
    entries: list[tuple[str, tuple[float, float, float]]],
    out_path: Path,
    cols: int = 1,
    cell_w: float = 2.0,
    cell_h: float = 0.2,
    pad_x: float = 0.1,
    pad_y: float = 0.1,
    font_size: int = 11,
    title: str | None = None,
    dpi: int = 200,
):
    """Draw a keypoint legend grid and save as PNG.

    Args:
        entries: List of (label, rgb) tuples where rgb is (r,g,b) in [0,1].
        out_path: Output PNG file path.
        cols: Number of columns in the grid.
        cell_w: Width of each cell (inches).
        cell_h: Height of each cell (inches).
        pad_x: Horizontal padding around the grid (inches).
        pad_y: Vertical padding around the grid (inches).
        font_size: Font size for labels.
        title: Optional title string to display at the top.
        dpi: Output image DPI.
    """
    n = len(entries)
    rows = math.ceil(n / cols)

    # Compute figure size (inches)
    fig_w = cols * cell_w + pad_x * 2
    fig_h = rows * cell_h + pad_y * 2 + (0.6 if title else 0)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    y_start = fig_h - pad_y
    if title:
        ax.text(
            fig_w / 2, y_start, title, ha="center", va="top", fontsize=font_size + 3, weight="bold"
        )
        y_start -= 0.5  # reduced space below title

    radius = 0.1
    text_x_offset = 0.2

    for idx, (label, rgb) in enumerate(entries):
        r = idx // cols
        c = idx % cols

        x0 = pad_x + c * cell_w
        y0 = y_start - r * cell_h

        # Color circle
        circle = plt.Circle(
            (x0 + radius, y0 - radius), radius, facecolor=rgb, edgecolor="black", linewidth=0.4
        )
        ax.add_patch(circle)

        # Label
        ax.text(
            x0 + 2 * radius + text_x_offset,
            y0 - radius,
            label,
            va="center",
            ha="left",
            fontsize=font_size,
        )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    """Parse args and generate the keypoint legend image."""
    p = argparse.ArgumentParser(description="Generate JABS keypoint legend image.")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG path, e.g., docs/images/keypoint-legend.png",
    )
    p.add_argument("--cols", type=int, default=1, help="Number of columns in the legend grid")
    p.add_argument("--font-size", type=int, default=9, help="Label font size")
    p.add_argument("--dpi", type=int, default=200, help="Output DPI")
    p.add_argument(
        "--title", type=str, default="JABS Keypoints", help="Legend title (set empty to omit)"
    )
    args = p.parse_args()

    entries = collect_entries()
    draw_legend(
        entries,
        out_path=args.out,
        cols=args.cols,
        font_size=args.font_size,
        dpi=args.dpi,
        title=(args.title if args.title else None),
    )


if __name__ == "__main__":
    main()
