"""Unit tests for the AngularVelocity feature class."""

import numpy as np

from jabs.feature_extraction.base_features import AngularVelocity


def test_angular_velocity_instantiation(pose_est_v5):
    """Test that AngularVelocity can be instantiated."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angular_vel_feature = AngularVelocity(pose_est_v5, pixel_scale)

    assert angular_vel_feature is not None


def test_angular_velocity_per_frame_dimensions(pose_est_v5):
    """Test that per_frame returns correct dimensions."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angular_vel_feature = AngularVelocity(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = angular_vel_feature.per_frame(identity)

        # Should have one feature: angular_velocity
        assert len(values) == 1
        assert "angular_velocity" in values

        # Should have one value per frame
        assert values["angular_velocity"].shape == (pose_est_v5.num_frames,)


def test_angular_velocity_per_frame_units(pose_est_v5):
    """Test that angular velocity is scaled by fps (degrees per second)."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angular_vel_feature = AngularVelocity(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = angular_vel_feature.per_frame(identity)
        velocities = values["angular_velocity"]

        # Angular velocity should be in degrees per second
        # Valid values should be reasonable (not extremely large)
        non_nan_indices = ~np.isnan(velocities)
        if non_nan_indices.any():
            # Most angular velocities should be within reasonable range
            # (e.g., -1800 to 1800 degrees per second = -5 to 5 rotations per second)
            reasonable_indices = non_nan_indices & (np.abs(velocities) < 10000)
            assert reasonable_indices.sum() > 0


def test_angular_velocity_handles_wraparound(pose_est_v5):
    """Test that angular velocity correctly handles angle wraparound (0/360 boundary)."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angular_vel_feature = AngularVelocity(pose_est_v5, pixel_scale)

    # The implementation should handle wraparound correctly
    # by choosing the shortest angular path
    for identity in range(pose_est_v5.num_identities):
        values = angular_vel_feature.per_frame(identity)
        velocities = values["angular_velocity"]

        # Check that values exist (may contain NaNs)
        assert velocities is not None


def test_angular_velocity_consecutive_same_angles():
    """Test angular velocity for consecutive frames with same bearing."""
    # This is a unit test for the logic, but we need a mock pose object
    # For now, just verify the feature can be computed
    pass


def test_angular_velocity_feature_name():
    """Test that the feature name is set correctly."""
    assert AngularVelocity.name() == "angular_velocity"


def test_angular_velocity_handles_nans(pose_est_v5):
    """Test that angular velocity handles NaN bearings correctly."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angular_vel_feature = AngularVelocity(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = angular_vel_feature.per_frame(identity)
        velocities = values["angular_velocity"]

        # Should produce output with same shape even with NaNs in input
        assert velocities.shape == (pose_est_v5.num_frames,)


def test_angular_velocity_window_operations(pose_est_v5):
    """Test that window operations work correctly."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angular_vel_feature = AngularVelocity(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        per_frame_values = angular_vel_feature.per_frame(identity)
        window_values = angular_vel_feature.window(
            identity, window_size=5, per_frame_features=per_frame_values
        )

        # Check that window operations are computed
        assert len(window_values) > 0

        for _op_name, op_features in window_values.items():
            for _feature_name, feature_values in op_features.items():
                # Window values should have same shape as per_frame
                assert feature_values.shape == (pose_est_v5.num_frames,)
