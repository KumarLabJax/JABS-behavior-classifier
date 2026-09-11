"""video reader

This package handles reading frames from a video file as well as applying various
annotations (animal track (trajectory), landmarks, identity markers).

Overlays that the GUI can toggle are drawn as Qt overlays instead, by
:mod:`jabs.overlay_drawing` and the player widget's overlay classes, so this package
stays free of Qt: it is imported by the process-pool workers in
``jabs.project.parallel_workers``.
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
