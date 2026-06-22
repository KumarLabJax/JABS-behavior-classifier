"""Tests for the shared pose-drawing helper used by the overlay and frame export."""

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    from jabs.pose_estimation import PoseEstimation
    from jabs.ui.player_widget.pose_drawing import draw_identity_pose, native_pose_sizes

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


class _StubPose:
    """Minimal stand-in for PoseEstimation exposing what draw_identity_pose needs."""

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
        (100, 100, (3, 2)),  # min bounds
        (800, 600, (3, 2)),
        (1920, 1080, (8, 5)),
    ],
    ids=["min", "sd", "hd"],
)
def test_native_pose_sizes(width, height, expected) -> None:
    """Keypoint/line sizes honor their minimums and scale with resolution."""
    assert native_pose_sizes(width, height) == expected


def test_native_pose_sizes_monotonic() -> None:
    """Larger frames never produce smaller markers."""
    small_kp, small_lw = native_pose_sizes(640, 480)
    large_kp, large_lw = native_pose_sizes(3840, 2160)
    assert large_kp > small_kp
    assert large_lw > small_lw


def test_draw_identity_pose_noop_when_points_missing() -> None:
    """Nothing is drawn when the identity has no pose at this frame."""
    painter = MagicMock()
    pose = _StubPose(len(PoseEstimation.KeypointIndex), points_present=False)

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
    num_keypoints = len(PoseEstimation.KeypointIndex)
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
    pose = _StubPose(len(PoseEstimation.KeypointIndex), points_present=True)

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
