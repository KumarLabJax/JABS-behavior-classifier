"""Tests for lixit distance feature extraction."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation
from jabs.feature_extraction.landmark_features import lixit
from jabs.pose_estimation import PoseEstimation


@pytest.fixture
def fresh_pose_v5():
    """A fresh PoseEstimationV5 with its own tempdir copy and no lixit set.

    Lets tests install custom lixit positions without mutating the shared,
    module-scoped ``pose_est_v5_with_static_objects`` fixture.

    Yields:
        PoseEstimationV5: pose estimation object with no static lixit object.
    """
    test_file = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v5.h5"
    with tempfile.TemporaryDirectory() as tmpdir:
        pose_path = Path(tmpdir) / "sample_pose_est_v5.h5"
        shutil.copy(test_file, pose_path)
        yield jabs.pose_estimation.open_pose_file(pose_path)


def test_dimensions(pose_est_v5_with_static_objects, lixit_distance_features):
    """Test dimensions of per frame and window feature values.

    Verifies that lixit distance features have the correct shape.
    """
    _, lixit_distance = lixit_distance_features

    # check dimensions of per frame feature values
    for i in range(pose_est_v5_with_static_objects.num_identities):
        distances = lixit_distance.per_frame(i)

        # Check that we got a dictionary back
        assert isinstance(distances, dict)

        # Check dimensions for NOSE keypoint (could check all keypoints)
        assert "distance to lixit NOSE" in distances
        assert distances["distance to lixit NOSE"].shape == (
            pose_est_v5_with_static_objects.num_frames,
        )

        # check dimensions of window feature values
        dist_window_values = lixit_distance.window(i, 5, distances)
        for op in dist_window_values:
            assert dist_window_values[op]["distance to lixit NOSE"].shape == (
                pose_est_v5_with_static_objects.num_frames,
            )


def test_distances_greater_equal_zero(pose_est_v5_with_static_objects, lixit_distance_features):
    """Test that computed distances are non-negative."""
    _, lixit_distance = lixit_distance_features

    for i in range(pose_est_v5_with_static_objects.num_identities):
        distances = lixit_distance.per_frame(i)

        # Check all distance features are >= 0 (ignoring NaN values)
        for feature_name, feature_values in distances.items():
            valid_distances = feature_values[~np.isnan(feature_values)]
            assert (valid_distances >= 0).all(), f"{feature_name} has negative distances"


def test_computation(lixit_distance_features):
    """Test lixit distance computation against expected values.

    Spot checks distance values for identity 0 against known results.
    These values have been verified to match manual Euclidean distance
    calculations between nose keypoint and lixit position [62, 166]
    with pixel scaling applied.
    """
    _, lixit_distance = lixit_distance_features

    # spot check some distance values for identity 0
    # Note: These expected values are for the NOSE keypoint distance to lixit
    # with lixit at position [62, 166] and pixel scale applied
    expected = np.asarray(
        [
            18.4560733207,
            18.5266151431,
            18.5773552151,
            18.6480151524,
            18.7383696105,
            18.7583096218,
            18.7583096218,
            18.8288618427,
            18.8288618427,
            18.9900097824,
        ],
        dtype=np.float32,
    )

    distances = lixit_distance.per_frame(0)
    actual = distances["distance to lixit NOSE"]

    # Compare first 10 values
    for i in range(expected.shape[0]):
        assert expected[i] == pytest.approx(actual[i], abs=1e-5)


def test_closest_lixit_per_frame_nearest_neighbor():
    """Gaps are filled from the temporally nearest valid frame, forward or backward.

    Valid choices occur at frame 1 (lixit 0) and frame 6 (lixit 2). The intervening
    gap splits by proximity to each neighbor; the leading gap is back-filled from
    frame 1 and the trailing gap is carried forward from frame 6.
    """
    nan = np.nan
    far = [9.0, 9.0, 9.0]

    def closest_to(idx: int) -> list[float]:
        row = list(far)
        row[idx] = 1.0
        return row

    alignment_distances = np.array(
        [
            [nan, nan, nan],  # 0 leading gap   -> back-fill from frame 1 -> 0
            closest_to(0),  # 1 valid          -> 0
            [nan, nan, nan],  # 2 gap (d1=1,d6=4) -> 0
            [nan, nan, nan],  # 3 gap (d1=2,d6=3) -> 0
            [nan, nan, nan],  # 4 gap (d1=3,d6=2) -> 2
            [nan, nan, nan],  # 5 gap (d1=4,d6=1) -> 2
            closest_to(2),  # 6 valid          -> 2
            [nan, nan, nan],  # 7 trailing gap   -> carry forward from frame 6 -> 2
        ],
        dtype=np.float32,
    )

    result = lixit.LixitDistanceInfo._closest_lixit_per_frame(alignment_distances)

    np.testing.assert_array_equal(result, np.array([0, 0, 0, 0, 2, 2, 2, 2], dtype=np.uint8))
    assert result.dtype == np.uint8


def test_closest_lixit_per_frame_tie_prefers_previous():
    """An equidistant gap frame is filled from the earlier valid frame."""
    nan = np.nan
    alignment_distances = np.array(
        [
            [1.0, 9.0],  # 0 valid -> 0
            [nan, nan],  # 1 equidistant to frames 0 and 2 -> tie -> previous -> 0
            [9.0, 1.0],  # 2 valid -> 1
        ],
        dtype=np.float32,
    )

    result = lixit.LixitDistanceInfo._closest_lixit_per_frame(alignment_distances)

    np.testing.assert_array_equal(result, np.array([0, 0, 1], dtype=np.uint8))


def test_closest_lixit_per_frame_all_nan():
    """When no frame has a centroid, every frame defaults to lixit 0."""
    alignment_distances = np.full((4, 3), np.nan, dtype=np.float32)
    result = lixit.LixitDistanceInfo._closest_lixit_per_frame(alignment_distances)
    np.testing.assert_array_equal(result, np.zeros(4, dtype=np.uint8))


def test_compute_centroids_handles_none_hull():
    """A None convex hull on a present frame yields NaN, not an exception."""
    from shapely.geometry import MultiPoint

    hull = MultiPoint([(0, 0), (2, 0), (1, 2)]).convex_hull

    class _StubPose:
        """Minimal pose stand-in exposing only what _compute_centroids needs."""

        num_frames = 3

        # frame 0 present with a hull, frame 1 present but no hull, frame 2 absent
        def identity_mask(self, identity: int) -> np.ndarray:
            return np.array([1, 1, 0], dtype=np.uint8)

        def get_identity_convex_hulls(self, identity: int) -> list:
            return [hull, None, None]

    info = lixit.LixitDistanceInfo(_StubPose(), pixel_scale=1.0)
    centroids = info.get_centroids(0)

    assert centroids.shape == (3, 2)
    assert not np.isnan(centroids[0]).any()  # valid hull -> finite centroid
    assert np.isnan(centroids[1]).all()  # None hull -> NaN, no crash
    assert np.isnan(centroids[2]).all()  # absent -> NaN


def test_centroid_drives_closest_selection(fresh_pose_v5):
    """The closest lixit is chosen by the mouse centroid, not the nose keypoint.

    Places one lixit on the nose and another on the centroid for a frame where the
    two are clearly separated. Centroid-based selection must pick the lixit on the
    centroid, whereas the previous nose-based logic would have picked the other.
    """
    pose = fresh_pose_v5
    pixel_scale = pose.cm_per_pixel
    info = lixit.LixitDistanceInfo(pose, pixel_scale)

    centroids = info.get_centroids(0)  # pixel units
    points, _ = pose.get_identity_poses(0)  # pixel units
    nose = points[:, PoseEstimation.KeypointIndex.NOSE, :]

    # find a frame where nose and centroid are both valid and clearly separated
    valid = ~np.isnan(centroids).any(axis=1) & ~np.isnan(nose).any(axis=1)
    separation = np.hypot(*(centroids - nose).T)
    candidates = np.where(valid & (separation > 5.0))[0]
    assert candidates.size > 0, "no frame with a clearly separated nose and centroid"
    frame = int(candidates[0])

    # lixit 0 sits on the nose, lixit 1 sits on the centroid
    pose.static_objects["lixit"] = np.asarray([nose[frame], centroids[frame]], dtype=np.float64)

    closest = info.get_closest_lixit(0)
    assert closest[frame] == 1


def test_mouse_lixit_angle_runs_with_three_point_lixit(fresh_pose_v5):
    """MouseLixitAngle produces in-range cosines using the shared centroid.

    Exercises the consolidated centroid path (reused from LixitDistanceInfo) for a
    three-keypoint lixit.
    """
    pose = fresh_pose_v5
    pixel_scale = pose.cm_per_pixel
    # three-keypoint lixit: tip, left side, right side
    pose.static_objects["lixit"] = np.asarray(
        [[[62, 166], [60, 170], [64, 170]]], dtype=np.float64
    )

    info = lixit.LixitDistanceInfo(pose, pixel_scale)
    feature = lixit.MouseLixitAngle(pose, pixel_scale, info)
    result = feature.per_frame(0)

    assert set(result) == {"centroid - nose", "base-tail - centroid"}
    for values in result.values():
        assert values.shape == (pose.num_frames,)
        finite = values[~np.isnan(values)]
        assert np.all((finite >= -1.0) & (finite <= 1.0))
