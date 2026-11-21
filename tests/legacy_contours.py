"""Test helper: pack contours padded the way a legacy pose file stores them.

Stand-in poses in ``tests/overlay_drawing`` and ``tests/video_export`` build
their segmentation windows with :func:`pack_segmentation`. It came from the
removed legacy v6 reader (``jabs/pose_estimation/pose_est_v6.py``); ``Pose``
reads already-packed contours from the unified file and has no equivalent.

Importable as a top-level module because pytest puts ``tests/`` (which has no
``__init__.py``) on ``sys.path``.
"""

import numpy as np
import numpy.typing as npt

from jabs.core.types import SegmentationWindow


def pack_segmentation(
    seg_data: npt.NDArray[np.integer],
    seg_flags: npt.NDArray[np.integer] | npt.NDArray[np.bool_] | None,
    start: int,
) -> SegmentationWindow:
    """Pack legacy ``-1``-padded contours so the valid ones come first.

    Uses the same rule as the legacy-to-unified converter: a contour slot holds a
    contour iff its first point's x is not -1, and the contour's points are the
    leading ones before the first -1 x. Valid contours keep their legacy order.

    Args:
        seg_data: One identity's contours, shape (n, C, P, 2), ``(x, y)``, padded
            with -1.
        seg_flags: Matching external flags, shape (n, C): positive for an
            external boundary, 0 for a hole, -1 for no contour. None without flags.
        start: Frame index of the first frame.

    Returns:
        The packed window; padding slots and points hold unspecified values.
    """
    num_points = seg_data.shape[2]
    padding = seg_data[..., 0] == -1
    exists = ~padding[..., 0]
    lengths = np.where(padding.any(axis=-1), padding.argmax(axis=-1), num_points)
    # stable, so valid contours keep their legacy order at the front
    order = np.argsort(~exists, axis=-1, kind="stable")
    packed_exists = np.take_along_axis(exists, order, axis=-1)
    contours = np.take_along_axis(seg_data, order[..., np.newaxis, np.newaxis], axis=1)
    contour_length = np.where(packed_exists, np.take_along_axis(lengths, order, axis=-1), 0)
    external_flag = None
    if seg_flags is not None:
        external_flag = np.take_along_axis(np.asarray(seg_flags) > 0, order, axis=-1)
    return SegmentationWindow(
        contours=contours.astype(np.int32),
        contour_count=exists.sum(axis=-1).astype(np.uint32),
        contour_length=contour_length.astype(np.uint32),
        external_flag=external_flag,
        start=start,
    )
