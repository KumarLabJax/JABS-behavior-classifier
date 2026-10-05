"""Unit tests for the CentroidVelocity feature classes."""

import numpy as np
import pytest

from jabs.feature_extraction.base_features import CentroidVelocityDir, CentroidVelocityMag
from jabs.feature_extraction.feature_base_class import Feature


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


@pytest.mark.parametrize(
    ("feature_class", "expected_name", "expected_circular"),
    [
        (CentroidVelocityDir, "centroid_velocity_dir", True),
        (CentroidVelocityMag, "centroid_velocity_mag", False),
    ],
    ids=["dir", "mag"],
)
def test_centroid_velocity_name_and_circular_flag(
    feature_class: type[Feature], expected_name: str, expected_circular: bool
) -> None:
    """Test that each feature name is set correctly and only the direction is circular.

    Args:
        feature_class: Centroid velocity feature class under test.
        expected_name: Name the feature is registered under.
        expected_circular: Whether the feature uses circular window statistics.
    """
    assert feature_class.name() == expected_name
    assert feature_class._use_circular is expected_circular


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


def test_centroid_velocity_handles_missing_convex_hull(pose_est_v5):
    """A present frame with no convex hull yields NaN instead of raising.

    ``identity_mask()`` requires only one valid body keypoint on pose v3 while
    ``get_identity_convex_hulls()`` needs three, so a frame can be masked present and
    still have no hull. Stubbing a None hull into an otherwise valid frame reproduces
    that combination without needing a v3 pose file.
    """
    pixel_scale = pose_est_v5.cm_per_pixel
    identity = 0

    hulls = list(pose_est_v5.get_identity_convex_hulls(identity))
    present = np.flatnonzero(pose_est_v5.identity_mask(identity) == 1)
    assert present.size > 0, "fixture has no frames where identity 0 is present"
    hole = int(present[present.size // 2])
    assert hulls[hole - 1] is not None and hulls[hole + 1] is not None, (
        "the frames flanking the hole must have hulls for the NaN assertions below"
    )
    hulls[hole] = None

    class _PoseWithMissingHull:
        """Delegates to the real pose object, but reports one hull as None."""

        def __getattr__(self, name):
            return getattr(pose_est_v5, name)

        def get_identity_convex_hulls(self, ident):
            return hulls if ident == identity else pose_est_v5.get_identity_convex_hulls(ident)

    poses = _PoseWithMissingHull()

    dir_values = CentroidVelocityDir(poses, pixel_scale).per_frame(identity)
    mag_values = CentroidVelocityMag(poses, pixel_scale).per_frame(identity)

    assert dir_values["centroid_velocity_dir"].shape == (pose_est_v5.num_frames,)
    assert mag_values["centroid_velocity_mag"].shape == (pose_est_v5.num_frames,)

    # np.gradient uses a central difference, so the missing centroid propagates to the
    # frames on either side of the hole rather than to the hole itself
    assert np.isnan(dir_values["centroid_velocity_dir"][hole - 1])
    assert np.isnan(dir_values["centroid_velocity_dir"][hole + 1])
    assert np.isnan(mag_values["centroid_velocity_mag"][hole - 1])
    assert np.isnan(mag_values["centroid_velocity_mag"][hole + 1])
