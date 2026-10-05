"""Unit tests for the Angles feature class."""

import numpy as np
import pytest

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


def test_angles_circular_window_operations_use_a_0_360_range() -> None:
    """Angles overrides the circular window operations to report angles in [0, 360).

    The default circular operations use [-180, 180), where the mean of 170 and 190
    degrees is -180. The Angles override reports it as 180.
    """
    mean = Angles._circular_window_operations["mean"]
    std_dev = Angles._circular_window_operations["std_dev"]

    assert mean(np.array([170.0, 190.0])) == pytest.approx(180.0)
    assert mean(np.array([10.0, 20.0])) == pytest.approx(15.0)
    assert std_dev(np.array([45.0, 45.0])) == pytest.approx(0.0, abs=1e-6)


def test_angles_window_means_are_never_negative(pose_est_v5_short) -> None:
    """Angles.window applies its [0, 360) circular mean, so no window mean is negative."""
    angles_feature = Angles(pose_est_v5_short, pose_est_v5_short.cm_per_pixel)

    for identity in range(pose_est_v5_short.num_identities):
        window_values = angles_feature.window(
            identity,
            window_size=5,
            per_frame_features=angles_feature.per_frame(identity),
        )

        # the sine and cosine columns use a plain mean and are legitimately negative
        angle_means = np.concatenate(
            [
                values
                for name, values in window_values["mean"].items()
                if not name.endswith(("sine", "cosine"))
            ]
        )
        assert np.nanmin(angle_means) >= 0.0
        assert np.nanmax(angle_means) < 360.0
        # the sample has angles past 180, which a [-180, 180) range would report as negative
        assert np.nanmax(angle_means) > 180.0


def test_angles_feature_name():
    """Test that the feature name is set correctly."""
    assert Angles.name() == "angles"


def test_angles_uses_circular_statistics():
    """Test that the Angles feature uses circular statistics."""
    assert Angles._use_circular is True
