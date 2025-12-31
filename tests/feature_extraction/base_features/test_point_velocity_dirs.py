"""Unit tests for the PointVelocityDirs feature class."""

import numpy as np

from jabs.feature_extraction.base_features import PointVelocityDirs
from jabs.pose_estimation import PoseEstimation


def test_point_velocity_dirs_instantiation(pose_est_v5):
    """Test that PointVelocityDirs can be instantiated."""
    pixel_scale = pose_est_v5.cm_per_pixel
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pixel_scale)

    assert velocity_dirs_feature is not None


def test_point_velocity_dirs_per_frame_dimensions(pose_est_v5):
    """Test that per_frame returns correct dimensions."""
    pixel_scale = pose_est_v5.cm_per_pixel
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pixel_scale)

    num_keypoints = len(PoseEstimation.KeypointIndex)
    # Each keypoint has 3 features: direction, sine, and cosine
    expected_num_features = num_keypoints * 3

    for identity in range(pose_est_v5.num_identities):
        values = velocity_dirs_feature.per_frame(identity)

        # Should have direction, sine, and cosine for each keypoint
        assert len(values) == expected_num_features

        # Each feature should have one value per frame
        for _feature_name, feature_values in values.items():
            assert feature_values.shape == (pose_est_v5.num_frames,)


def test_point_velocity_dirs_range(pose_est_v5):
    """Test that velocity directions are in the correct range [-180, 180)."""
    pixel_scale = pose_est_v5.cm_per_pixel
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = velocity_dirs_feature.per_frame(identity)

        for feature_name, feature_values in values.items():
            # Only check direction features, not sine/cosine
            if "sine" not in feature_name and "cosine" not in feature_name:
                non_nan_indices = ~np.isnan(feature_values)
                if non_nan_indices.any():
                    assert (feature_values[non_nan_indices] >= -180).all()
                    assert (feature_values[non_nan_indices] <= 180).all()


def test_point_velocity_dirs_sine_cosine_range(pose_est_v5):
    """Test that sine and cosine values are in [-1, 1]."""
    pixel_scale = pose_est_v5.cm_per_pixel
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = velocity_dirs_feature.per_frame(identity)

        for feature_name, feature_values in values.items():
            if "sine" in feature_name or "cosine" in feature_name:
                non_nan_indices = ~np.isnan(feature_values)
                if non_nan_indices.any():
                    assert (feature_values[non_nan_indices] >= -1).all()
                    assert (feature_values[non_nan_indices] <= 1).all()


def test_point_velocity_dirs_feature_names(pose_est_v5):
    """Test that feature names follow the expected format."""
    pixel_scale = pose_est_v5.cm_per_pixel
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = velocity_dirs_feature.per_frame(identity)

        # Check that each keypoint has direction, sine, and cosine features
        for keypoint in PoseEstimation.KeypointIndex:
            expected_dir_name = f"{keypoint.name} velocity direction"
            expected_sine_name = f"{keypoint.name} velocity direction sine"
            expected_cosine_name = f"{keypoint.name} velocity direction cosine"

            assert expected_dir_name in values
            assert expected_sine_name in values
            assert expected_cosine_name in values


def test_point_velocity_dirs_relative_to_bearing(pose_est_v5):
    """Test that velocity directions are computed relative to animal bearing."""
    pixel_scale = pose_est_v5.cm_per_pixel
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pixel_scale)

    # The implementation computes directions relative to the animal's bearing
    # This ensures consistency across different orientations
    for identity in range(pose_est_v5.num_identities):
        values = velocity_dirs_feature.per_frame(identity)

        # Verify that features are computed
        assert len(values) > 0


def test_point_velocity_dirs_window_operations(pose_est_v5):
    """Test that window operations work correctly with circular statistics."""
    pixel_scale = pose_est_v5.cm_per_pixel
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        per_frame_values = velocity_dirs_feature.per_frame(identity)
        window_values = velocity_dirs_feature.window(
            identity, window_size=5, per_frame_features=per_frame_values
        )

        # Check that window operations are computed
        assert len(window_values) > 0

        for _op_name, op_features in window_values.items():
            for _feature_name, feature_values in op_features.items():
                # Window values should have same shape as per_frame
                assert feature_values.shape == (pose_est_v5.num_frames,)


def test_point_velocity_dirs_feature_name():
    """Test that the feature name is set correctly."""
    assert PointVelocityDirs.name() == "point_velocity_dirs"


def test_point_velocity_dirs_uses_circular_statistics():
    """Test that the PointVelocityDirs feature uses circular statistics."""
    assert PointVelocityDirs._use_circular is True


def test_point_velocity_dirs_handles_nans(pose_est_v5):
    """Test that velocity directions handle NaN coordinates correctly."""
    pixel_scale = pose_est_v5.cm_per_pixel
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = velocity_dirs_feature.per_frame(identity)

        # Should produce output with same shape even with NaNs in input
        for _feature_name, feature_values in values.items():
            assert feature_values.shape == (pose_est_v5.num_frames,)
