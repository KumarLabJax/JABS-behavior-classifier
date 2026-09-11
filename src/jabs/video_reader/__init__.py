"""video reader

This package handles reading frames from a video file as well as applying various
annotations (animal track (trajectory), landmarks, identity markers).

The pose, label, identity and segmentation overlays are drawn with Qt instead, by
:mod:`jabs.overlay_drawing` and the player widget's overlay classes. The annotations
still drawn here are baked into the frame as it is decoded, and the GUI toggles them
the same way. Whatever remains in this package has to stay free of Qt: it is imported
by the process-pool workers in ``jabs.project.parallel_workers``.
"""

from .frame_annotation import (
    draw_track,
    mark_identity,
    overlay_landmarks,
)
from .video_reader import VideoReader

__all__ = [
    "VideoReader",
    "draw_track",
    "mark_identity",
    "overlay_landmarks",
]
