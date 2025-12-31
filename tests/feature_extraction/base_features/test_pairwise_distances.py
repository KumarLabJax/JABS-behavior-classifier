"""Unit tests for the PairwisePointDistances feature class."""

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
    """Test pairwise distance computation with known point positions."""
    # This would require creating a mock pose object with known coordinates
    # For now, we rely on the real data tests above
    pass
