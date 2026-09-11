"""Shared Qt drawing for the overlays JABS paints on a video frame.

Three kinds of overlay live here, all drawn the same way by every caller:

* the pose skeleton (:func:`draw_identity_pose`), drawn by the on-screen
  :class:`~jabs.ui.player_widget.overlays.pose_overlay.PoseOverlay` at the scaled and
  cropped display resolution, by the full-resolution frame export, and by the overlay
  video export in :mod:`jabs.video_export`.
* the per-identity behavior label/prediction marker (:func:`draw_label_marker`), drawn
  by the on-screen :class:`~jabs.ui.player_widget.overlays.label_overlay.LabelOverlay`
  and by the same video export.
* the per-identity segmentation contours (:func:`draw_identity_segmentation`), drawn by
  the on-screen
  :class:`~jabs.ui.player_widget.overlays.segmentation_overlay.SegmentationOverlay` and
  by both exports.

The only thing that differs between callers is how image coordinates map to the
painter's coordinate space, so that mapping is passed in as ``to_output``, and how
large the markers should be, which :func:`native_overlay_scale` derives from the frame
size for the native-resolution exports.

This is its own top-level package rather than living inside an existing one on
purpose, and the alternatives are all worse:

* :mod:`jabs.ui` - ``jabs/ui/__init__.py`` imports ``MainWindow``, so anything
  importing from under ``jabs.ui`` drags in the whole GUI. That made
  :mod:`jabs.video_export` circular and would force the CLI to import the
  application just to draw a skeleton.
* :mod:`jabs.video_reader` - the obvious neighbour, since it already owns
  ``frame_annotation``. But that package is imported by
  ``jabs.project.parallel_workers``, which runs in process-pool workers, so putting
  Qt behind it would add Qt's import cost to every worker spawn.
* :mod:`jabs.video_export` - the on-screen overlays use this too, so the GUI would
  end up importing the *export* package to draw its live view.
* :mod:`jabs.utils` - that is a thin re-export shim for ``jabs-core``'s update-check
  helpers, not a general utility package. Putting Qt and ``distinctipy`` behind it
  would make ``from jabs.utils import check_for_update`` an order of magnitude more
  expensive for every caller.
"""

from .colors import (
    BACKGROUND_COLOR,
    BEHAVIOR_COLOR,
    KEYPOINT_COLOR_MAP,
    NOT_BEHAVIOR_COLOR,
)
from .labels import (
    LABEL_MARKER_SIZE,
    draw_label_marker,
    label_marker_color,
    native_label_marker_sizes,
)
from .scaling import native_overlay_scale
from .segmentation import (
    SEGMENTATION_ACTIVE_COLOR,
    SEGMENTATION_INACTIVE_COLOR,
    draw_identity_segmentation,
    identity_contours,
    native_segmentation_line_width,
)
from .skeleton import (
    KEYPOINT_SIZE,
    LINE_SEGMENT_COLOR,
    draw_identity_pose,
    native_pose_sizes,
)

__all__ = [
    "BACKGROUND_COLOR",
    "BEHAVIOR_COLOR",
    "KEYPOINT_COLOR_MAP",
    "KEYPOINT_SIZE",
    "LABEL_MARKER_SIZE",
    "LINE_SEGMENT_COLOR",
    "NOT_BEHAVIOR_COLOR",
    "SEGMENTATION_ACTIVE_COLOR",
    "SEGMENTATION_INACTIVE_COLOR",
    "draw_identity_pose",
    "draw_identity_segmentation",
    "draw_label_marker",
    "identity_contours",
    "label_marker_color",
    "native_label_marker_sizes",
    "native_overlay_scale",
    "native_pose_sizes",
    "native_segmentation_line_width",
]
