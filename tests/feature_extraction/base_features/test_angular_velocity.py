"""Unit tests for the AngularVelocity feature class."""

from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt

from jabs.feature_extraction.base_features import AngularVelocity

_FPS = 30.0


def _pose_with_bearings(bearings: npt.NDArray[np.float64]) -> MagicMock:
    """Build a mock pose whose animal has the given per-frame bearings.

    Args:
        bearings: Bearing of the animal in degrees for each frame.

    Returns:
        Mock pose reporting ``bearings`` for every identity at ``_FPS`` frames per second.
    """
    mock_pose = MagicMock()
    mock_pose.num_frames = len(bearings)
    mock_pose.num_identities = 1
    mock_pose.fps = _FPS
    mock_pose.compute_all_bearings.return_value = bearings
    return mock_pose


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


def test_angular_velocity_per_frame_units() -> None:
    """Test that angular velocity is in degrees per second, not degrees per frame."""
    # The bearing turns 1 degree every frame, so at 30 frames per second it turns 30 deg/s.
    bearings = np.arange(10, dtype=np.float64)

    velocities = AngularVelocity(_pose_with_bearings(bearings), 1.0).per_frame(0)[
        "angular_velocity"
    ]

    # the last frame has no following frame to compare against
    np.testing.assert_allclose(velocities, [_FPS] * 9 + [np.nan])


def test_angular_velocity_handles_wraparound() -> None:
    """Test that angular velocity takes the shortest path across the 0/360 boundary."""
    # 350 -> 10 is a 20 degree turn forward and 10 -> 350 is 20 degrees back, not +/-340.
    bearings = np.array([350.0, 10.0, 350.0])

    velocities = AngularVelocity(_pose_with_bearings(bearings), 1.0).per_frame(0)[
        "angular_velocity"
    ]

    np.testing.assert_allclose(velocities, [20.0 * _FPS, -20.0 * _FPS, np.nan])


def test_angular_velocity_consecutive_same_angles():
    """Test angular velocity for consecutive frames with same bearing.

    Creates a mock pose object where the animal maintains the same bearing
    angle across consecutive frames, verifying that angular velocity is zero.
    """
    # Create a mock pose with constant bearing
    num_frames = 10

    mock_pose = MagicMock()
    mock_pose.num_frames = num_frames
    mock_pose.num_identities = 1
    mock_pose.fps = 30.0  # 30 frames per second

    # Mock compute_all_bearings to return constant bearing angle of 45 degrees
    # All frames have the same bearing, so angular velocity should be zero
    constant_bearings = np.full(num_frames, 45.0, dtype=np.float32)
    mock_pose.compute_all_bearings.return_value = constant_bearings

    # Create feature instance
    angular_vel_feature = AngularVelocity(mock_pose, 1.0)

    # Get computed angular velocities
    values = angular_vel_feature.per_frame(0)
    velocities = values["angular_velocity"]

    # Filter out NaN values (first frame typically has NaN)
    non_nan_velocities = velocities[~np.isnan(velocities)]

    # Angular velocity should be zero (or very close to zero) for constant bearing
    # Allow small numerical errors
    np.testing.assert_array_almost_equal(
        non_nan_velocities,
        np.zeros_like(non_nan_velocities),
        decimal=3,
        err_msg=f"Constant bearing should give zero angular velocity, got {non_nan_velocities}",
    )


def test_angular_velocity_feature_name():
    """Test that the feature name is set correctly."""
    assert AngularVelocity.name() == "angular_velocity"


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
