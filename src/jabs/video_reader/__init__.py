"""video reader

This package reads frames from a video file. That is all it does: everything JABS
draws on a frame is painted by :mod:`jabs.overlay_drawing` and the player widget's
overlay classes, over the frame rather than into it.

Keeping the drawing out has a practical reason beyond tidiness - this package is
imported by the process-pool workers in ``jabs.project.parallel_workers``, so anything
behind it is paid for on every worker spawn. Qt in particular must not end up here.
"""

from .video_reader import VideoReader

__all__ = [
    "VideoReader",
]
