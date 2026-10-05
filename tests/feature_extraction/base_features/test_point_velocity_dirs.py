"""Unit tests for the PointVelocityDirs feature class."""

import numpy as np
import pytest

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


def _pose_with_constant_motion(pose_est_v5, delta: tuple[float, float]) -> object:
    """Wrap a pose object so every keypoint translates by ``delta`` each frame.

    Bearings are reported as a constant 0 degrees (animal facing +x), so the
    bearing-relative velocity direction depends only on ``delta``. Delegating to the
    real pose object keeps the attributes the feature does not control (fps, frame
    count, keypoint layout) realistic.

    Args:
        pose_est_v5: Real pose object to delegate unhandled attributes to.
        delta: Per-frame ``(dx, dy)`` translation applied to every keypoint.

    Returns:
        A pose-like object suitable for passing to :class:`PointVelocityDirs`.
    """
    num_frames = pose_est_v5.num_frames
    num_keypoints = len(PoseEstimation.KeypointIndex)

    frame_numbers = np.arange(num_frames, dtype=np.float32)[:, np.newaxis]
    poses = np.empty((num_frames, num_keypoints, 2), dtype=np.float32)
    poses[:, :, 0] = frame_numbers * delta[0]
    poses[:, :, 1] = frame_numbers * delta[1]
    point_masks = np.ones((num_frames, num_keypoints), dtype=np.uint8)
    bearings = np.zeros(num_frames, dtype=np.float32)

    class _PoseWithConstantMotion:
        """Delegates to the real pose object, but reports synthetic poses/bearings."""

        def __getattr__(self, name):
            return getattr(pose_est_v5, name)

        def get_identity_poses(self, identity, pixel_scale=None):
            return poses, point_masks

        def compute_all_bearings(self, identity):
            return bearings

    return _PoseWithConstantMotion()


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ((1.0, 0.0), -180.0),
        ((0.0, 1.0), -90.0),
        ((-1.0, 0.0), 0.0),
        ((0.0, -1.0), 90.0),
    ],
    ids=["along-bearing", "90-off-bearing", "against-bearing", "270-off-bearing"],
)
def test_point_velocity_dirs_bearing_relative_convention(pose_est_v5, delta, expected):
    """Pin the bearing-relative angle convention this feature emits.

    With the animal facing +x (bearing 0) and every keypoint translating by ``delta``
    each frame, the velocity direction relative to the bearing is fully determined.

    These expectations encode the 180 degree offset documented on
    :class:`PointVelocityDirs`: motion straight along the bearing reads ``-180``
    rather than ``0``. Correcting the offset changes cached feature values, so it
    needs a ``FEATURE_VERSION`` bump - and updating these expectations by 180 degrees.
    """
    poses = _pose_with_constant_motion(pose_est_v5, delta)
    values = PointVelocityDirs(poses, 1.0).per_frame(0)

    for keypoint in PoseEstimation.KeypointIndex:
        direction = values[f"{keypoint.name} velocity direction"]
        assert direction == pytest.approx(expected), keypoint.name

        # sine/cosine must stay consistent with the direction they are derived from
        assert values[f"{keypoint.name} velocity direction sine"] == pytest.approx(
            np.sin(np.deg2rad(expected)), abs=1e-6
        )
        assert values[f"{keypoint.name} velocity direction cosine"] == pytest.approx(
            np.cos(np.deg2rad(expected)), abs=1e-6
        )


@pytest.mark.parametrize(
    ("delta", "intended"),
    [
        ((1.0, 0.0), 0.0),
        ((0.0, 1.0), 90.0),
        ((0.0, -1.0), -90.0),
    ],
    ids=["along-bearing", "90-off-bearing", "270-off-bearing"],
)
def test_point_velocity_dirs_offset_is_a_constant_rotation(pose_est_v5, delta, intended):
    """Rotating the emitted angles by 180 degrees recovers the intended convention.

    This records that the gap against ``CentroidVelocityDir`` is one constant rotation
    rather than a sporadic discrepancy, so whoever fixes the wrap can confirm it closes
    the gap for every direction at once.
    """
    poses = _pose_with_constant_motion(pose_est_v5, delta)
    values = PointVelocityDirs(poses, 1.0).per_frame(0)

    for keypoint in PoseEstimation.KeypointIndex:
        emitted = values[f"{keypoint.name} velocity direction"]
        # undo the offset, then wrap back into [-180, 180) the way
        # jabs.core.utils.geometry.signed_angle_degrees does
        rotated = ((emitted + 180.0) + 180.0) % 360.0 - 180.0
        assert rotated == pytest.approx(intended), keypoint.name


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
