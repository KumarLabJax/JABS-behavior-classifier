"""Render JABS pose overlays onto video frames and write annotated videos.

Split so the single-frame renderer can be used on its own - the GUI's "Export
Frame" needs only :func:`render_overlay_frame`, while the video exports also pull
in :func:`export_overlay_video`.
"""

from .frame_renderer import render_overlay_frame
from .video_writer import DEFAULT_CODEC, VideoExportError, export_overlay_video

__all__ = [
    "DEFAULT_CODEC",
    "VideoExportError",
    "export_overlay_video",
    "render_overlay_frame",
]
