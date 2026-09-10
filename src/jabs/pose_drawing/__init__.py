"""Shared pose-skeleton drawing.

Three callers draw the same skeleton with :func:`draw_identity_pose`: the on-screen
:class:`~jabs.ui.player_widget.overlays.pose_overlay.PoseOverlay` (at the scaled and
cropped display resolution), the full-resolution frame export, and the overlay video
export in :mod:`jabs.video_export`. The only thing that differs is how image
coordinates map to the painter's coordinate space, so that mapping is passed in as
``to_output``.

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
* :mod:`jabs.video_export` - the on-screen overlay uses this too, so the GUI would
  end up importing the *export* package to draw its live view.
* :mod:`jabs.utils` - that is a thin re-export shim for ``jabs-core``'s update-check
  helpers, not a general utility package. Putting Qt and ``distinctipy`` behind it
  would make ``from jabs.utils import check_for_update`` an order of magnitude more
  expensive for every caller.
"""

from .colors import KEYPOINT_COLOR_MAP
from .skeleton import (
    KEYPOINT_SIZE,
    LINE_SEGMENT_COLOR,
    draw_identity_pose,
    native_pose_sizes,
)

__all__ = [
    "KEYPOINT_COLOR_MAP",
    "KEYPOINT_SIZE",
    "LINE_SEGMENT_COLOR",
    "draw_identity_pose",
    "native_pose_sizes",
]
