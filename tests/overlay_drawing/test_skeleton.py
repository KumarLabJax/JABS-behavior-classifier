"""Tests for the shared pose-drawing helper used by the overlay and frame export."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from jabs.core.pose_skeleton import KeypointIndex

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    from jabs.overlay_drawing import draw_identity_pose, native_pose_sizes

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


class _StubPose:
    """Minimal stand-in for PoseLike exposing what draw_identity_pose needs."""

    def __init__(self, num_keypoints: int, *, points_present: bool) -> None:
        self._n = num_keypoints
        self._present = points_present

    def get_points(self, frame_index: int, identity: int):
        """Return zeroed points with an all-present (or no) mask."""
        if not self._present:
            return None, None
        points = np.zeros((self._n, 2), dtype=np.float32)
        mask = np.ones(self._n, dtype=np.uint8)
        return points, mask

    def get_connected_segments(self):
        """No skeleton segments, so only keypoint circles are drawn."""
        return []


def _identity_transform(x, y):
    return int(x), int(y)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (100, 100, (3, 2)),  # below the reference size, so the floors apply
        (800, 600, (3, 2)),  # larger dimension is 800, so the scale is exactly 1
        (800, 800, (3, 2)),  # the footage the base sizes are calibrated on
        (1920, 1080, (5, 3)),
        (3840, 2160, (7, 4)),
    ],
    ids=["min", "sd", "reference", "hd", "4k"],
)
def test_native_pose_sizes(width, height, expected) -> None:
    """Keypoint/line sizes honor their minimums and scale with resolution."""
    assert native_pose_sizes(width, height) == expected


def test_native_pose_sizes_grow_sub_linearly() -> None:
    """Markers must grow more slowly than the frame.

    Linear growth keeps a marker at a fixed share of the frame, which produced dots
    heavy enough to cover the animal on high-resolution footage. Quadrupling the
    frame should grow the marker by less than four times.
    """
    base_kp, base_lw = native_pose_sizes(800, 800)
    quad_kp, quad_lw = native_pose_sizes(3200, 3200)

    assert quad_kp > base_kp, "markers should still grow with resolution"
    assert quad_kp < base_kp * 4, "growth must be sub-linear"
    assert quad_lw < base_lw * 4


def test_native_pose_sizes_monotonic() -> None:
    """Larger frames never produce smaller markers."""
    small_kp, small_lw = native_pose_sizes(640, 480)
    large_kp, large_lw = native_pose_sizes(3840, 2160)
    assert large_kp > small_kp
    assert large_lw > small_lw


def test_draw_identity_pose_noop_when_points_missing() -> None:
    """Nothing is drawn when the identity has no pose at this frame."""
    painter = MagicMock()
    pose = _StubPose(len(KeypointIndex), points_present=False)

    draw_identity_pose(
        painter,
        pose,
        frame_index=0,
        identity=0,
        to_output=_identity_transform,
        keypoint_size=4,
        line_width=2,
        active=True,
    )

    painter.drawEllipse.assert_not_called()
    painter.drawLine.assert_not_called()


def test_draw_identity_pose_draws_each_visible_keypoint() -> None:
    """One keypoint circle is drawn per present, in-bounds keypoint."""
    painter = MagicMock()
    num_keypoints = len(KeypointIndex)
    pose = _StubPose(num_keypoints, points_present=True)

    draw_identity_pose(
        painter,
        pose,
        frame_index=0,
        identity=0,
        to_output=_identity_transform,
        keypoint_size=4,
        line_width=2,
        active=True,
    )

    assert painter.drawEllipse.call_count == num_keypoints


def test_draw_identity_pose_skips_points_filtered_by_transform() -> None:
    """Keypoints whose transform returns None (e.g. outside a crop) are not drawn."""
    painter = MagicMock()
    pose = _StubPose(len(KeypointIndex), points_present=True)

    draw_identity_pose(
        painter,
        pose,
        frame_index=0,
        identity=0,
        to_output=lambda x, y: None,
        keypoint_size=4,
        line_width=2,
        active=True,
    )

    painter.drawEllipse.assert_not_called()


@pytest.mark.parametrize("kind", ["legacy", "unified"])
def test_pose_overlay_paints_a_real_pose(segmentation_poses, kind) -> None:
    """The player's pose overlay paints the float32 points a real Pose returns.

    Qt's ``QRect.contains`` rejects ``np.float32`` coordinates with OverflowError, so
    passing a Pose's points through unconverted broke every repaint of the player.
    """
    from PySide6 import QtCore, QtGui
    from PySide6.QtWidgets import QApplication

    from jabs.pose_estimation import open_pose_file
    from jabs.ui.player_widget.frame_with_overlays import FrameWithOverlaysWidget
    from jabs.ui.player_widget.overlays.pose_overlay import PoseOverlay

    _app = QApplication.instance() or QApplication([])
    pose = open_pose_file(segmentation_poses[kind])
    widget = FrameWithOverlaysWidget()
    widget.set_pose(pose)
    widget.update_frame(QtGui.QImage(800, 800, QtGui.QImage.Format.Format_RGB888), 0)
    widget.pose_overlay_mode = FrameWithOverlaysWidget.PoseOverlayMode.ALL
    widget.set_active_identity(0)
    # what paintEvent records for an unscaled, uncropped 800x800 frame
    widget._scaled_pix_x = widget._scaled_pix_y = 0
    widget._scaled_pix_width = widget._scaled_pix_height = 800
    overlay = next(o for o in widget.overlays if isinstance(o, PoseOverlay))

    image = QtGui.QImage(800, 800, QtGui.QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QtGui.QPainter(image)
    try:
        overlay.paint(painter, QtCore.QRect(0, 0, 800, 800))
    finally:
        painter.end()

    # the skeletons were drawn
    drawn = sum(image.pixel(x, y) != 0 for x in range(800) for y in range(800))
    assert drawn > 0
