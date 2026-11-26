"""Tests for the shared landmark distance/bearing cache accessors."""

import numpy as np
import numpy.typing as npt
import pytest

import jabs.feature_extraction.landmark_features.corner as corner_module
import jabs.feature_extraction.landmark_features.lixit as lixit_module
from jabs.feature_extraction.landmark_features.distance_info import LandmarkDistanceInfo


class _CountingInfo(LandmarkDistanceInfo):
    """Minimal subclass that records how often the cache was filled."""

    def __init__(self) -> None:
        super().__init__(poses=None, pixel_scale=1.0)
        self.fill_count = 0

    def cache_features(self, identity: int) -> None:
        """Fill one identity's columns, returning early when already cached."""
        if identity in self._cached_distances and identity in self._cached_bearings:
            return
        self.fill_count += 1
        self._cached_distances[identity] = {"distance": np.zeros(3, dtype=np.float32)}
        self._cached_bearings[identity] = {"bearing": np.ones(3, dtype=np.float32)}


def test_accessors_fill_the_cache_once() -> None:
    """The first access computes; later accesses reuse what was computed."""
    info = _CountingInfo()

    assert info.get_distances(0)["distance"].shape == (3,)
    assert info.get_bearings(0)["bearing"].shape == (3,)
    assert info.fill_count == 1

    info.get_distances(0)
    info.get_bearings(0)
    assert info.fill_count == 1

    # a second identity is computed on its own
    info.get_bearings(1)
    assert info.fill_count == 2


@pytest.mark.parametrize(
    "accessor", ["get_distances", "get_bearings"], ids=["distances", "bearings"]
)
def test_accessors_do_not_hand_out_the_cached_dict(accessor: str) -> None:
    """A caller adding a column to the result cannot change what the next caller gets."""
    info = _CountingInfo()

    first = getattr(info, accessor)(0)
    first["derived"] = np.full(3, 2.0, dtype=np.float32)

    second = getattr(info, accessor)(0)
    assert "derived" not in second
    assert first is not second
    # the value arrays themselves are still shared, not copied
    for name, values in second.items():
        assert values is first[name]


def test_corner_bearing_feature_leaves_the_cache_alone(pose_est_v5_with_static_objects) -> None:
    """BearingToCorner adds sine/cosine columns to its own result, not to the cache."""
    poses = pose_est_v5_with_static_objects
    pixel_scale = poses.cm_per_pixel
    info = corner_module.CornerDistanceInfo(poses, pixel_scale)
    feature = corner_module.BearingToCorner(poses, pixel_scale, info)

    first = feature.per_frame(0)
    assert "bearing to corner cosine" in first
    assert set(info.get_bearings(0)) == {"bearing to corner", "bearing to center"}

    # the feature's own output is unchanged by a repeat call
    second = feature.per_frame(0)
    assert set(second) == set(first)
    for name, values in second.items():
        _assert_same_values(values, first[name])


def test_lixit_bearing_feature_leaves_the_cache_alone(pose_est_v5_with_static_objects) -> None:
    """BearingToLixit adds sine/cosine columns to its own result, not to the cache."""
    poses = pose_est_v5_with_static_objects
    pixel_scale = poses.cm_per_pixel
    info = lixit_module.LixitDistanceInfo(poses, pixel_scale)
    feature = lixit_module.BearingToLixit(poses, pixel_scale, info)

    first = feature.per_frame(0)
    assert "bearing to lixit sine" in first
    assert set(info.get_bearings(0)) == {"bearing to lixit"}

    second = feature.per_frame(0)
    assert set(second) == set(first)
    for name, values in second.items():
        _assert_same_values(values, first[name])


def _assert_same_values(
    actual: npt.NDArray[np.floating], expected: npt.NDArray[np.floating]
) -> None:
    """Assert two per-frame columns match, treating nan as equal to nan."""
    np.testing.assert_array_equal(actual, expected)
