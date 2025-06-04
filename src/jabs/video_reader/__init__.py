"""video reader

This package handles reading frames from a video file as well as applying various
annotations (such as pose overlay, animal track (trajectory), segmentation, etc.)
"""

from .frame_annotation import (
    draw_track,
    label_identity,
    overlay_landmarks,
    overlay_pose,
    overlay_segmentation,
)
from .video_reader import VideoReader

__all__ = [
    "VideoReader",
    "draw_track",
    "label_identity",
    "overlay_landmarks",
    "overlay_pose",
    "overlay_segmentation",
]
