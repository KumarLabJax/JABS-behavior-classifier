"""Tests for the pose accessors shared by the identity-indexed pose readers."""

import numpy as np
import numpy.typing as npt
import pytest

from jabs.pose_estimation import (
    PoseEstimationV3,
    PoseEstimationV4,
    PoseEstimationV5,
    PoseEstimationV6,
    PoseEstimationV7,
    PoseEstimationV8,
)
from jabs.pose_estimation.identity_indexed_pose import IdentityIndexedPoseMixin

_NUM_IDENTITIES = 2
_NUM_FRAMES = 3
_NUM_KEYPOINTS = 12


class _StubPose(IdentityIndexedPoseMixin):
    """Minimal holder of identity-indexed pose arrays, for exercising the mixin."""

    def __init__(
        self,
        points: npt.NDArray[np.float64],
        point_mask: npt.NDArray[np.bool_],
        identity_mask: npt.NDArray[np.bool_],
    ) -> None:
        self._points = points
        self._point_mask = point_mask
        self._identity_mask = identity_mask


@pytest.fixture
def stub_pose() -> _StubPose:
    """A two identity, three frame pose with identity 1 absent from the last frame."""
    points = np.arange(
        _NUM_IDENTITIES * _NUM_FRAMES * _NUM_KEYPOINTS * 2, dtype=np.float64
    ).reshape(_NUM_IDENTITIES, _NUM_FRAMES, _NUM_KEYPOINTS, 2)
    point_mask = np.ones((_NUM_IDENTITIES, _NUM_FRAMES, _NUM_KEYPOINTS), dtype=np.bool_)
    # the tip of the tail is never observed for any identity in any frame
    point_mask[..., _NUM_KEYPOINTS - 1] = False
    identity_mask = np.ones((_NUM_IDENTITIES, _NUM_FRAMES), dtype=np.bool_)
    identity_mask[1, _NUM_FRAMES - 1] = False
    return _StubPose(points, point_mask, identity_mask)


def test_get_points_returns_frame_slice(stub_pose: _StubPose) -> None:
    """get_points returns the points and mask for one identity in one frame."""
    points, mask = stub_pose.get_points(1, 0)
    assert np.array_equal(points, stub_pose._points[0, 1])
    assert np.array_equal(mask, stub_pose._point_mask[0, 1])


def test_get_points_applies_scale(stub_pose: _StubPose) -> None:
    """A scale factor converts the points, leaving the mask untouched."""
    points, mask = stub_pose.get_points(1, 0, scale=0.5)
    assert np.array_equal(points, stub_pose._points[0, 1] * 0.5)
    # the mask is not a distance, so scaling must leave it alone
    assert np.array_equal(mask, stub_pose._point_mask[0, 1])


def test_get_points_absent_identity(stub_pose: _StubPose) -> None:
    """A frame the identity is absent from yields no points and no mask."""
    assert stub_pose.get_points(_NUM_FRAMES - 1, 1) == (None, None)


def test_get_identity_poses_returns_all_frames(stub_pose: _StubPose) -> None:
    """get_identity_poses returns every frame for the requested identity."""
    points, mask = stub_pose.get_identity_poses(1)
    assert points.shape == (_NUM_FRAMES, _NUM_KEYPOINTS, 2)
    assert np.array_equal(points, stub_pose._points[1])
    assert np.array_equal(mask, stub_pose._point_mask[1])


def test_get_identity_poses_applies_scale(stub_pose: _StubPose) -> None:
    """A scale factor converts every frame's points, leaving the mask untouched."""
    points, mask = stub_pose.get_identity_poses(1, scale=2.0)
    assert np.array_equal(points, stub_pose._points[1] * 2.0)
    assert np.array_equal(mask, stub_pose._point_mask[1])


def test_identity_mask_is_per_frame(stub_pose: _StubPose) -> None:
    """identity_mask reports presence for each frame of one identity."""
    assert np.array_equal(stub_pose.identity_mask(1), [True, True, False])


def test_get_identity_point_mask(stub_pose: _StubPose) -> None:
    """get_identity_point_mask returns every frame's mask for one identity."""
    assert np.array_equal(stub_pose.get_identity_point_mask(0), stub_pose._point_mask[0])


def test_get_reduced_point_mask(stub_pose: _StubPose) -> None:
    """A keypoint never observed by any identity is excluded from the reduced mask."""
    reduced = stub_pose.get_reduced_point_mask()
    expected = np.ones(_NUM_KEYPOINTS, dtype=np.bool_)
    expected[_NUM_KEYPOINTS - 1] = False
    assert np.array_equal(reduced, expected)


_ACCESSORS = (
    "get_points",
    "get_identity_poses",
    "identity_mask",
    "get_identity_point_mask",
    "get_reduced_point_mask",
)


@pytest.mark.parametrize(
    "reader",
    [
        PoseEstimationV3,
        PoseEstimationV4,
        PoseEstimationV5,
        PoseEstimationV6,
        PoseEstimationV7,
        PoseEstimationV8,
    ],
    ids=["v3", "v4", "v5", "v6", "v7", "v8"],
)
def test_readers_share_one_implementation(reader: type) -> None:
    """Every v3+ reader resolves these accessors to the mixin, not to a copy of it.

    Args:
        reader: Pose reader class under test.
    """
    for method in _ACCESSORS:
        assert getattr(reader, method) is getattr(IdentityIndexedPoseMixin, method), (
            f"{reader.__name__}.{method} is not the IdentityIndexedPoseMixin implementation"
        )
