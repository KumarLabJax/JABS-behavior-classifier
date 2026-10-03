"""Unit tests for jabs.core.utils.pose_util module."""

import numpy as np
import pytest
from shapely.geometry import MultiPoint

from jabs.core.utils.pose_util import identity_centroids


class _StubPose:
    """Minimal pose stand-in exposing only what ``identity_centroids`` reads.

    Args:
        mask: Per-frame identity mask, one entry per frame.
        hulls: Per-frame convex hulls, ``None`` where no hull could be built.
    """

    def __init__(self, mask: np.ndarray, hulls: list) -> None:
        self.num_frames = len(mask)
        self._mask = mask
        self._hulls = hulls

    def identity_mask(self, identity: int) -> np.ndarray:
        """Return the per-frame identity mask."""
        return self._mask

    def get_identity_convex_hulls(self, identity: int) -> list:
        """Return the per-frame convex hulls."""
        return self._hulls


@pytest.fixture
def square_hull():
    """A unit square hull centered on (0.5, 0.5)."""
    return MultiPoint([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]).convex_hull


def test_identity_centroids_returns_hull_center(square_hull) -> None:
    """A present frame with a hull gets that hull's centroid."""
    poses = _StubPose(np.array([1], dtype=np.uint8), [square_hull])

    centroids = identity_centroids(poses, 0)

    assert centroids.shape == (1, 2)
    assert centroids.dtype == np.float32
    np.testing.assert_allclose(centroids[0], [0.5, 0.5])


def test_identity_centroids_absent_frame_is_nan(square_hull) -> None:
    """A frame the identity is absent from is NaN even if a hull is present."""
    poses = _StubPose(np.array([0], dtype=np.uint8), [square_hull])

    centroids = identity_centroids(poses, 0)

    assert np.isnan(centroids[0]).all()


def test_identity_centroids_present_frame_without_hull_is_nan(square_hull) -> None:
    """A present frame with no convex hull is NaN rather than raising.

    ``identity_mask()`` needs only one valid body keypoint on pose v3 while a convex
    hull needs three, so a frame can be masked present and still have no hull.
    """
    poses = _StubPose(np.array([1, 1, 0], dtype=np.uint8), [square_hull, None, None])

    centroids = identity_centroids(poses, 0)

    assert centroids.shape == (3, 2)
    np.testing.assert_allclose(centroids[0], [0.5, 0.5])
    assert np.isnan(centroids[1]).all()
    assert np.isnan(centroids[2]).all()


def test_identity_centroids_no_frames() -> None:
    """A pose with no frames yields an empty (0, 2) array."""
    poses = _StubPose(np.array([], dtype=np.uint8), [])

    centroids = identity_centroids(poses, 0)

    assert centroids.shape == (0, 2)
