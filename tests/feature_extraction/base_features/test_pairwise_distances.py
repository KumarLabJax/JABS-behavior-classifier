"""Unit tests for the PairwisePointDistances feature class."""

from unittest.mock import MagicMock

import numpy as np

from jabs.feature_extraction.base_features import PairwisePointDistances
from jabs.pose_estimation import PoseEstimation


def test_pairwise_distances_instantiation(pose_est_v5):
    """Test that PairwisePointDistances can be instantiated."""
    pixel_scale = pose_est_v5.cm_per_pixel
    pairwise_feature = PairwisePointDistances(pose_est_v5, pixel_scale)

    assert pairwise_feature is not None


def test_pairwise_distances_per_frame_dimensions(pose_est_v5):
    """Test that per_frame returns correct dimensions."""
    pixel_scale = pose_est_v5.cm_per_pixel
    pairwise_feature = PairwisePointDistances(pose_est_v5, pixel_scale)

    num_keypoints = len(PoseEstimation.KeypointIndex)
    # Number of unique pairs is n*(n-1)/2
    expected_num_pairs = num_keypoints * (num_keypoints - 1) // 2

    for identity in range(pose_est_v5.num_identities):
        values = pairwise_feature.per_frame(identity)

        # Should have one feature per unique pair of keypoints
        assert len(values) == expected_num_pairs

        # Each feature should have one value per frame
        for _feature_name, feature_values in values.items():
            assert feature_values.shape == (pose_est_v5.num_frames,)


def test_pairwise_distances_non_negative(pose_est_v5):
    """Test that all distances are non-negative."""
    pixel_scale = pose_est_v5.cm_per_pixel
    pairwise_feature = PairwisePointDistances(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = pairwise_feature.per_frame(identity)

        for _feature_name, feature_values in values.items():
            non_nan_indices = ~np.isnan(feature_values)
            if non_nan_indices.any():
                assert (feature_values[non_nan_indices] >= 0).all()


def test_pairwise_distances_symmetry(pose_est_v5):
    """Test that distance(A, B) == distance(B, A) implicitly through naming."""
    pixel_scale = pose_est_v5.cm_per_pixel
    pairwise_feature = PairwisePointDistances(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = pairwise_feature.per_frame(identity)

        # Check that each pair appears only once (no duplicates like "A-B" and "B-A")
        feature_names = list(values.keys())
        reversed_pairs = ["-".join(reversed(name.split("-"))) for name in feature_names]

        for reversed_pair in reversed_pairs:
            # The reversed pair should not be in the feature names
            assert reversed_pair not in feature_names


def test_pairwise_distances_feature_names(pose_est_v5):
    """Test that feature names follow the expected format."""
    pixel_scale = pose_est_v5.cm_per_pixel
    pairwise_feature = PairwisePointDistances(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = pairwise_feature.per_frame(identity)

        for feature_name in values:
            # Each feature name should be in the format "keypoint1-keypoint2"
            assert "-" in feature_name
            parts = feature_name.split("-")
            assert len(parts) == 2


def test_pairwise_distances_scaled_correctly(pose_est_v5):
    """Test that distances are scaled by pixel_scale."""
    pixel_scale = pose_est_v5.cm_per_pixel
    pairwise_feature = PairwisePointDistances(pose_est_v5, pixel_scale)

    # Get distances with scaling
    for identity in range(pose_est_v5.num_identities):
        values = pairwise_feature.per_frame(identity)

        # Check that distances are in reasonable range for scaled values
        # (assuming pixel_scale is in cm/pixel, distances should be in cm)
        for _feature_name, feature_values in values.items():
            non_nan_indices = ~np.isnan(feature_values)
            if non_nan_indices.any():
                # Distances should be positive and reasonable
                assert (feature_values[non_nan_indices] >= 0).all()


def test_pairwise_distances_window_operations(pose_est_v5):
    """Test that window operations work correctly."""
    pixel_scale = pose_est_v5.cm_per_pixel
    pairwise_feature = PairwisePointDistances(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        per_frame_values = pairwise_feature.per_frame(identity)
        window_values = pairwise_feature.window(
            identity, window_size=5, per_frame_features=per_frame_values
        )

        # Check that window operations are computed
        assert len(window_values) > 0

        for _op_name, op_features in window_values.items():
            for _feature_name, feature_values in op_features.items():
                # Window values should have same shape as per_frame
                assert feature_values.shape == (pose_est_v5.num_frames,)


def test_pairwise_distances_feature_name():
    """Test that the feature name is set correctly."""
    assert PairwisePointDistances.name() == "pairwise_distances"


def test_pairwise_distances_known_values():
    """Test pairwise distance computation with known point positions.

    Creates a mock pose object with known keypoint positions and verifies
    that computed distances match manual calculations.
    """
    # Create a simple scenario with 2 keypoints and 3 frames
    # Frame 0: NOSE at (0, 0), BASE_NECK at (3, 4) -> distance = 5
    # Frame 1: NOSE at (0, 0), BASE_NECK at (6, 8) -> distance = 10
    # Frame 2: NOSE at (1, 1), BASE_NECK at (4, 5) -> distance = 5

    mock_pose = MagicMock(spec=PoseEstimation)
    mock_pose.num_frames = 3
    mock_pose.num_identities = 1
    mock_pose.cm_per_pixel = 1.0  # No scaling for simplicity

    # Create points array: shape (num_frames, num_keypoints, 2)
    # Only populate NOSE and BASE_NECK, rest can be NaN
    num_keypoints = len(PoseEstimation.KeypointIndex)
    points = np.full((3, num_keypoints, 2), np.nan, dtype=np.float32)

    # Set NOSE positions (keypoint index 0)
    points[0, PoseEstimation.KeypointIndex.NOSE, :] = [0, 0]
    points[1, PoseEstimation.KeypointIndex.NOSE, :] = [0, 0]
    points[2, PoseEstimation.KeypointIndex.NOSE, :] = [1, 1]

    # Set BASE_NECK positions (keypoint index 1)
    points[0, PoseEstimation.KeypointIndex.BASE_NECK, :] = [3, 4]
    points[1, PoseEstimation.KeypointIndex.BASE_NECK, :] = [6, 8]
    points[2, PoseEstimation.KeypointIndex.BASE_NECK, :] = [4, 5]

    # Create point mask (all valid for our two keypoints)
    point_mask = np.zeros((3, num_keypoints), dtype=bool)
    point_mask[:, PoseEstimation.KeypointIndex.NOSE] = True
    point_mask[:, PoseEstimation.KeypointIndex.BASE_NECK] = True

    # Mock the get_identity_poses method
    mock_pose.get_identity_poses.return_value = (points, point_mask)

    # Create feature instance
    pairwise_feature = PairwisePointDistances(mock_pose, 1.0)

    # Get computed distances
    values = pairwise_feature.per_frame(0)

    # Find the NOSE-BASE_NECK distance feature
    nose_neck_key = None
    for key in values:
        if "NOSE" in key and "BASE_NECK" in key:
            nose_neck_key = key
            break

    assert nose_neck_key is not None, "NOSE-BASE_NECK feature not found"

    # Verify the computed distances match expected values
    computed_distances = values[nose_neck_key]
    expected_distances = np.array([5.0, 10.0, 5.0], dtype=np.float32)

    np.testing.assert_array_almost_equal(
        computed_distances,
        expected_distances,
        decimal=5,
        err_msg=f"Computed distances {computed_distances} do not match expected {expected_distances}",
    )
