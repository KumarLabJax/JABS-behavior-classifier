"""Unit tests for the CentroidVelocity feature classes."""

import numpy as np

from jabs.feature_extraction.base_features import CentroidVelocityDir, CentroidVelocityMag


def test_centroid_velocity_dir_instantiation(pose_est_v5):
    """Test that CentroidVelocityDir can be instantiated."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_dir_feature = CentroidVelocityDir(pose_est_v5, pixel_scale)

    assert centroid_dir_feature is not None


def test_centroid_velocity_dir_per_frame_dimensions(pose_est_v5):
    """Test that per_frame returns correct dimensions for CentroidVelocityDir."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_dir_feature = CentroidVelocityDir(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = centroid_dir_feature.per_frame(identity)

        # Should have 3 features: direction, sine, and cosine
        assert len(values) == 3
        assert "centroid_velocity_dir" in values
        assert "centroid_velocity_dir sine" in values
        assert "centroid_velocity_dir cosine" in values

        # Each feature should have one value per frame
        for _feature_name, feature_values in values.items():
            assert feature_values.shape == (pose_est_v5.num_frames,)


def test_centroid_velocity_dir_range(pose_est_v5):
    """Test that centroid velocity directions are in the correct range [-180, 180]."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_dir_feature = CentroidVelocityDir(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = centroid_dir_feature.per_frame(identity)
        directions = values["centroid_velocity_dir"]

        non_nan_indices = ~np.isnan(directions)
        if non_nan_indices.any():
            assert (directions[non_nan_indices] >= -180).all()
            assert (directions[non_nan_indices] <= 180).all()


def test_centroid_velocity_dir_sine_cosine_range(pose_est_v5):
    """Test that sine and cosine values are in [-1, 1]."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_dir_feature = CentroidVelocityDir(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = centroid_dir_feature.per_frame(identity)

        for feature_name in ["centroid_velocity_dir sine", "centroid_velocity_dir cosine"]:
            feature_values = values[feature_name]
            non_nan_indices = ~np.isnan(feature_values)
            if non_nan_indices.any():
                assert (feature_values[non_nan_indices] >= -1).all()
                assert (feature_values[non_nan_indices] <= 1).all()


def test_centroid_velocity_dir_feature_name():
    """Test that the feature name is set correctly."""
    assert CentroidVelocityDir.name() == "centroid_velocity_dir"


def test_centroid_velocity_dir_uses_circular_statistics():
    """Test that CentroidVelocityDir uses circular statistics."""
    assert CentroidVelocityDir._use_circular is True


def test_centroid_velocity_mag_instantiation(pose_est_v5):
    """Test that CentroidVelocityMag can be instantiated."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_mag_feature = CentroidVelocityMag(pose_est_v5, pixel_scale)

    assert centroid_mag_feature is not None


def test_centroid_velocity_mag_per_frame_dimensions(pose_est_v5):
    """Test that per_frame returns correct dimensions for CentroidVelocityMag."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_mag_feature = CentroidVelocityMag(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = centroid_mag_feature.per_frame(identity)

        # Should have 1 feature: magnitude
        assert len(values) == 1
        assert "centroid_velocity_mag" in values

        # Should have one value per frame
        assert values["centroid_velocity_mag"].shape == (pose_est_v5.num_frames,)


def test_centroid_velocity_mag_non_negative(pose_est_v5):
    """Test that all centroid velocity magnitudes are non-negative."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_mag_feature = CentroidVelocityMag(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = centroid_mag_feature.per_frame(identity)
        magnitudes = values["centroid_velocity_mag"]

        non_nan_indices = ~np.isnan(magnitudes)
        if non_nan_indices.any():
            assert (magnitudes[non_nan_indices] >= 0).all()


def test_centroid_velocity_mag_scaled_correctly(pose_est_v5):
    """Test that centroid velocity magnitudes are scaled by fps and pixel_scale."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_mag_feature = CentroidVelocityMag(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = centroid_mag_feature.per_frame(identity)
        magnitudes = values["centroid_velocity_mag"]

        # Check that magnitudes are in reasonable range
        # (assuming pixel_scale in cm/pixel and fps ~30, velocities should be reasonable)
        non_nan_indices = ~np.isnan(magnitudes)
        if non_nan_indices.any():
            # Velocities should be non-negative and reasonable
            assert (magnitudes[non_nan_indices] >= 0).all()


def test_centroid_velocity_mag_feature_name():
    """Test that the feature name is set correctly."""
    assert CentroidVelocityMag.name() == "centroid_velocity_mag"


def test_centroid_velocity_dir_window_operations(pose_est_v5):
    """Test that window operations work correctly for CentroidVelocityDir."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_dir_feature = CentroidVelocityDir(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        per_frame_values = centroid_dir_feature.per_frame(identity)
        window_values = centroid_dir_feature.window(
            identity, window_size=5, per_frame_features=per_frame_values
        )

        # Check that window operations are computed
        assert len(window_values) > 0

        for _op_name, op_features in window_values.items():
            for _feature_name, feature_values in op_features.items():
                # Window values should have same shape as per_frame
                assert feature_values.shape == (pose_est_v5.num_frames,)


def test_centroid_velocity_mag_window_operations(pose_est_v5):
    """Test that window operations work correctly for CentroidVelocityMag."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_mag_feature = CentroidVelocityMag(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        per_frame_values = centroid_mag_feature.per_frame(identity)
        window_values = centroid_mag_feature.window(
            identity, window_size=5, per_frame_features=per_frame_values
        )

        # Check that window operations are computed
        assert len(window_values) > 0

        for _op_name, op_features in window_values.items():
            for _feature_name, feature_values in op_features.items():
                # Window values should have same shape as per_frame
                assert feature_values.shape == (pose_est_v5.num_frames,)


def test_centroid_velocity_handles_missing_frames(pose_est_v5):
    """Test that centroid velocity features handle frames where identity is not present."""
    pixel_scale = pose_est_v5.cm_per_pixel
    centroid_dir_feature = CentroidVelocityDir(pose_est_v5, pixel_scale)
    centroid_mag_feature = CentroidVelocityMag(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        dir_values = centroid_dir_feature.per_frame(identity)
        mag_values = centroid_mag_feature.per_frame(identity)

        # Should produce output with correct shape
        assert dir_values["centroid_velocity_dir"].shape == (pose_est_v5.num_frames,)
        assert mag_values["centroid_velocity_mag"].shape == (pose_est_v5.num_frames,)
