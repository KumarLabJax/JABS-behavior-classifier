"""Tests for estimating the frame size of a pose file opened without its video."""

from unittest import mock

import numpy as np
import pytest

try:
    from jabs.ui.viewer.frame_size import estimate_frame_size

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def _fake_pose(identity_points: list[np.ndarray], static_objects: dict | None = None) -> mock.Mock:
    """Build a stand-in pose whose identities have the given (x, y) points, all valid."""
    pose = mock.Mock()
    pose.identities = list(range(len(identity_points)))
    pose.get_identity_poses.side_effect = lambda i: (
        identity_points[i],
        np.ones(identity_points[i].shape[:-1], dtype=np.uint8),
    )
    pose.static_objects = static_objects or {}
    return pose


def test_size_rounds_up_to_contain_the_furthest_points() -> None:
    """Width comes from x and height from y, each rounded up to a multiple of 100."""
    points = np.array([[[10.0, 20.0], [712.4, 350.0]]])
    pose = _fake_pose([points])

    assert estimate_frame_size(pose) == (800, 400)


def test_arena_landmarks_extend_the_frame() -> None:
    """Arena corners usually sit near the frame edge, so they count too."""
    points = np.array([[[100.0, 100.0]]])
    corners = np.array([[5, 5], [790, 5], [790, 795], [5, 795]])
    pose = _fake_pose([points], {"corners": corners})

    assert estimate_frame_size(pose) == (800, 800)


def test_masked_points_are_ignored() -> None:
    """Invalid keypoints (often stored as 0 or garbage) do not size the frame."""
    pose = mock.Mock()
    pose.identities = [0]
    pose.get_identity_poses.return_value = (
        np.array([[[150.0, 150.0], [5000.0, 5000.0]]]),
        np.array([[1, 0]], dtype=np.uint8),
    )
    pose.static_objects = {}

    assert estimate_frame_size(pose) == (200, 200)


def test_point_on_a_boundary_gets_a_larger_frame() -> None:
    """A point at x=800 is outside an 800 pixel wide frame (columns 0-799)."""
    pose = _fake_pose([np.array([[[800.0, 799.9]]])])

    assert estimate_frame_size(pose) == (900, 800)


def test_pose_without_points_gets_the_minimum_size() -> None:
    """With nothing to measure the frame is still large enough to display."""
    pose = _fake_pose([])

    assert estimate_frame_size(pose) == (100, 100)
