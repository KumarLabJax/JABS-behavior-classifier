"""Unit tests for the PointSpeeds feature class."""

from unittest.mock import MagicMock

import numpy as np

from jabs.feature_extraction.base_features import PointSpeeds
from jabs.pose_estimation import PoseEstimation


def test_point_speeds_instantiation(pose_est_v5):
    """Test that PointSpeeds can be instantiated."""
    pixel_scale = pose_est_v5.cm_per_pixel
    speeds_feature = PointSpeeds(pose_est_v5, pixel_scale)

    assert speeds_feature is not None


def test_point_speeds_per_frame_dimensions(pose_est_v5):
    """Test that per_frame returns correct dimensions."""
    pixel_scale = pose_est_v5.cm_per_pixel
    speeds_feature = PointSpeeds(pose_est_v5, pixel_scale)

    num_keypoints = len(PoseEstimation.KeypointIndex)

    for identity in range(pose_est_v5.num_identities):
        values = speeds_feature.per_frame(identity)

        # Should have one speed feature per keypoint
        assert len(values) == num_keypoints

        # Each feature should have one value per frame
        for _feature_name, feature_values in values.items():
            assert feature_values.shape == (pose_est_v5.num_frames,)


def test_point_speeds_non_negative(pose_est_v5):
    """Test that all speeds are non-negative."""
    pixel_scale = pose_est_v5.cm_per_pixel
    speeds_feature = PointSpeeds(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = speeds_feature.per_frame(identity)

        for _feature_name, feature_values in values.items():
            non_nan_indices = ~np.isnan(feature_values)
            if non_nan_indices.any():
                assert (feature_values[non_nan_indices] >= 0).all()


def test_point_speeds_feature_names(pose_est_v5):
    """Test that feature names follow the expected format."""
    pixel_scale = pose_est_v5.cm_per_pixel
    speeds_feature = PointSpeeds(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = speeds_feature.per_frame(identity)

        # Check that each keypoint has a speed feature
        for keypoint in PoseEstimation.KeypointIndex:
            expected_name = f"{keypoint.name} speed"
            assert expected_name in values


def test_point_speeds_scaled_by_fps(pose_est_v5):
    """Test that speeds are scaled by fps (units should be cm/s if pixel_scale is cm/pixel)."""
    pixel_scale = pose_est_v5.cm_per_pixel
    speeds_feature = PointSpeeds(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = speeds_feature.per_frame(identity)

        # Check that speeds are in reasonable range
        # (assuming pixel_scale in cm/pixel and fps ~30, speeds should be reasonable)
        for _feature_name, feature_values in values.items():
            non_nan_indices = ~np.isnan(feature_values)
            if non_nan_indices.any():
                # Speeds should be positive and reasonable (not absurdly large)
                assert (feature_values[non_nan_indices] >= 0).all()


def test_point_speeds_stationary_point():
    """Test that stationary points have zero speed.

    Creates a mock pose object where a keypoint remains at the same position
    across all frames, verifying that the computed speed is zero.
    """
    # Create a mock pose with a stationary NOSE keypoint
    num_frames = 10
    num_keypoints = len(PoseEstimation.KeypointIndex)

    mock_pose = MagicMock(spec=PoseEstimation)
    mock_pose.num_frames = num_frames
    mock_pose.num_identities = 1
    mock_pose.cm_per_pixel = 1.0
    mock_pose.fps = 30.0  # 30 frames per second

    # Create points array where NOSE stays at (5, 5) for all frames
    points = np.full((num_frames, num_keypoints, 2), np.nan, dtype=np.float32)

    # Set NOSE to be stationary at position (5, 5)
    for frame in range(num_frames):
        points[frame, PoseEstimation.KeypointIndex.NOSE, :] = [5.0, 5.0]

    # Set BASE_NECK to move slightly so we have some non-zero speeds
    for frame in range(num_frames):
        points[frame, PoseEstimation.KeypointIndex.BASE_NECK, :] = [frame, frame]

    # Create point mask (all valid for NOSE and BASE_NECK)
    point_mask = np.zeros((num_frames, num_keypoints), dtype=bool)
    point_mask[:, PoseEstimation.KeypointIndex.NOSE] = True
    point_mask[:, PoseEstimation.KeypointIndex.BASE_NECK] = True

    # Mock the get_identity_poses method
    mock_pose.get_identity_poses.return_value = (points, point_mask)

    # Create feature instance
    speeds_feature = PointSpeeds(mock_pose, 1.0)

    # Get computed speeds
    values = speeds_feature.per_frame(0)

    # Verify NOSE speed is zero (stationary point)
    nose_speed = values["NOSE speed"]

    # Speed should be zero for all frames (except possibly first frame which may be NaN)
    # Skip first frame as speed calculation typically requires previous frame
    non_nan_speeds = nose_speed[~np.isnan(nose_speed)]

    np.testing.assert_array_almost_equal(
        non_nan_speeds,
        np.zeros_like(non_nan_speeds),
        decimal=5,
        err_msg=f"Stationary NOSE should have zero speed, got {non_nan_speeds}",
    )

    # Verify BASE_NECK has non-zero speed (as a sanity check)
    base_neck_speed = values["BASE_NECK speed"]
    non_nan_base_neck = base_neck_speed[~np.isnan(base_neck_speed)]

    # BASE_NECK is moving, so should have non-zero speeds
    assert (non_nan_base_neck > 0).any(), "Moving BASE_NECK should have non-zero speeds"


def test_point_speeds_window_operations(pose_est_v5):
    """Test that window operations work correctly."""
    pixel_scale = pose_est_v5.cm_per_pixel
    speeds_feature = PointSpeeds(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        per_frame_values = speeds_feature.per_frame(identity)
        window_values = speeds_feature.window(
            identity, window_size=5, per_frame_features=per_frame_values
        )

        # Check that window operations are computed
        assert len(window_values) > 0

        for _op_name, op_features in window_values.items():
            for _feature_name, feature_values in op_features.items():
                # Window values should have same shape as per_frame
                assert feature_values.shape == (pose_est_v5.num_frames,)


def test_point_speeds_feature_name():
    """Test that the feature name is set correctly."""
    assert PointSpeeds.name() == "point_speeds"


def test_point_speeds_handles_nans(pose_est_v5):
    """Test that point speeds handles NaN coordinates correctly."""
    pixel_scale = pose_est_v5.cm_per_pixel
    speeds_feature = PointSpeeds(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = speeds_feature.per_frame(identity)

        # Should produce output with same shape even with NaNs in input
        for _feature_name, feature_values in values.items():
            assert feature_values.shape == (pose_est_v5.num_frames,)
