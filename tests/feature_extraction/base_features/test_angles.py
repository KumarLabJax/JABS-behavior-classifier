"""Unit tests for the Angles feature class."""

import numpy as np

from jabs.feature_extraction.angle_index import AngleIndex
from jabs.feature_extraction.base_features import Angles


def test_angles_instantiation(pose_est_v5):
    """Test that Angles can be instantiated with a pose estimation object."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angles_feature = Angles(pose_est_v5, pixel_scale)

    assert angles_feature is not None
    assert angles_feature._num_angles == len(AngleIndex)


def test_angles_per_frame_dimensions(pose_est_v5):
    """Test that per_frame returns correct dimensions."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angles_feature = Angles(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = angles_feature.per_frame(identity)

        # Should have 3 features per angle: angle, sine, and cosine
        expected_num_features = len(AngleIndex) * 3
        assert len(values) == expected_num_features

        # Each feature should have one value per frame
        for _feature_name, feature_values in values.items():
            assert feature_values.shape == (pose_est_v5.num_frames,)


def test_angles_per_frame_range(pose_est_v5):
    """Test that angles are in the correct range [0, 360)."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angles_feature = Angles(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = angles_feature.per_frame(identity)

        for feature_name, feature_values in values.items():
            # Only check angle features, not sine/cosine
            if "sine" not in feature_name and "cosine" not in feature_name:
                non_nan_indices = ~np.isnan(feature_values)
                if non_nan_indices.any():
                    assert (feature_values[non_nan_indices] >= 0).all()
                    assert (feature_values[non_nan_indices] < 360).all()


def test_angles_sine_cosine_range(pose_est_v5):
    """Test that sine and cosine values are in [-1, 1]."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angles_feature = Angles(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        values = angles_feature.per_frame(identity)

        for feature_name, feature_values in values.items():
            if "sine" in feature_name or "cosine" in feature_name:
                non_nan_indices = ~np.isnan(feature_values)
                if non_nan_indices.any():
                    assert (feature_values[non_nan_indices] >= -1).all()
                    assert (feature_values[non_nan_indices] <= 1).all()


def test_angles_compute_angles_basic():
    """Test the static _compute_angles method with known values."""
    # Test angle computation
    a = np.array([[0, 0]])
    b = np.array([[1, 0]])
    c = np.array([[1, 1]])

    angles = Angles._compute_angles(a, b, c)
    assert angles.shape == (1,)
    # The implementation computes arctan2(c-b) - arctan2(a-b) then mod 360
    # For these points: arctan2((1,0)) - arctan2((0,-1)) = 90 - 180 = -90 = 270 (mod 360)
    np.testing.assert_allclose(angles[0], 270.0, rtol=1e-5)


def test_angles_compute_angles_straight_line():
    """Test _compute_angles with collinear points."""
    # Test a straight line (180 degrees)
    a = np.array([[0, 0]])
    b = np.array([[1, 0]])
    c = np.array([[2, 0]])

    angles = Angles._compute_angles(a, b, c)
    assert angles.shape == (1,)
    # The angle should be 180 degrees
    np.testing.assert_allclose(angles[0], 180.0, rtol=1e-5)


def test_angles_compute_angles_multiple_points():
    """Test _compute_angles with multiple points."""
    # Multiple angles at once - test with collinear points and simple cases
    a = np.array([[0, 0], [0, 0], [1, 0]])
    b = np.array([[1, 0], [1, 0], [0, 0]])
    c = np.array([[1, 1], [2, 0], [0, 1]])

    angles = Angles._compute_angles(a, b, c)
    assert angles.shape == (3,)

    # Verify all angles are in [0, 360) range
    assert np.all(angles >= 0)
    assert np.all(angles < 360)


def test_angles_window_operations(pose_est_v5):
    """Test that window operations work correctly with circular statistics."""
    pixel_scale = pose_est_v5.cm_per_pixel
    angles_feature = Angles(pose_est_v5, pixel_scale)

    for identity in range(pose_est_v5.num_identities):
        per_frame_values = angles_feature.per_frame(identity)
        window_values = angles_feature.window(
            identity, window_size=5, per_frame_features=per_frame_values
        )

        # Check that window operations are computed
        assert len(window_values) > 0

        for _op_name, op_features in window_values.items():
            for _feature_name, feature_values in op_features.items():
                # Window values should have same shape as per_frame
                assert feature_values.shape == (pose_est_v5.num_frames,)


def test_angles_feature_name():
    """Test that the feature name is set correctly."""
    assert Angles.name() == "angles"


def test_angles_uses_circular_statistics():
    """Test that the Angles feature uses circular statistics."""
    assert Angles._use_circular is True
