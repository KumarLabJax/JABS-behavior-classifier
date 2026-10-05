"""Unit tests for the PointVelocityDirs feature class."""

import typing
from typing import Protocol

import numpy as np
import numpy.typing as npt
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


class _PoseLike(Protocol):
    """The slice of the pose interface ``PointVelocityDirs`` consumes.

    ``Feature.__init__`` reads ``fps``; ``PointVelocityDirs.per_frame`` calls the two
    methods. Spelling it out keeps the stub below honest about what it has to provide.
    """

    @property
    def fps(self) -> int:
        """Frames per second of the source video."""
        ...

    def get_identity_poses(
        self, identity: int, pixel_scale: float | None = None
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Return ``(poses, point_masks)`` for one identity."""
        ...

    def compute_all_bearings(self, identity: int) -> npt.NDArray[np.float32]:
        """Return the per-frame bearing of one identity, in degrees."""
        ...


def _pose_with_constant_motion(
    pose_est_v5: PoseEstimation,
    delta: tuple[float, float],
    bearing: float,
) -> _PoseLike:
    """Wrap a pose object so every keypoint translates by ``delta`` each frame.

    The bearing is reported as a constant, so the bearing-relative velocity direction
    is fully determined by ``delta`` and ``bearing`` and the expected angle can be
    written down exactly. Delegating to the real pose object keeps the attributes the
    feature does not control (fps, frame count, keypoint layout) realistic.

    Args:
        pose_est_v5: Real pose object to delegate unhandled attributes to.
        delta: Per-frame ``(dx, dy)`` translation applied to every keypoint. Its
            direction is the global motion direction, ``degrees(atan2(dy, dx))``.
        bearing: Constant bearing to report for every frame, in degrees.

    Returns:
        A pose-like object suitable for passing to :class:`PointVelocityDirs`.
    """
    num_frames: int = pose_est_v5.num_frames
    num_keypoints: int = len(PoseEstimation.KeypointIndex)

    frame_numbers = np.arange(num_frames, dtype=np.float32)[:, np.newaxis]
    poses: npt.NDArray[np.float32] = np.empty((num_frames, num_keypoints, 2), dtype=np.float32)
    poses[:, :, 0] = frame_numbers * delta[0]
    poses[:, :, 1] = frame_numbers * delta[1]
    point_masks: npt.NDArray[np.uint8] = np.ones((num_frames, num_keypoints), dtype=np.uint8)
    bearings: npt.NDArray[np.float32] = np.full(num_frames, bearing, dtype=np.float32)

    class _PoseWithConstantMotion:
        """Delegates to the real pose object, but reports synthetic poses/bearings."""

        def __getattr__(self, name: str) -> object:
            return getattr(pose_est_v5, name)

        def get_identity_poses(
            self, identity: int, pixel_scale: float | None = None
        ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
            """Return the synthetic constant-motion poses, for any identity."""
            return poses, point_masks

        def compute_all_bearings(self, identity: int) -> npt.NDArray[np.float32]:
            """Return the constant bearing, for any identity."""
            return bearings

    return typing.cast("_PoseLike", _PoseWithConstantMotion())


# (delta, bearing, intended, emitted): `intended` is the bearing-relative angle the
# feature is meant to report, `emitted` is what it actually reports. The cases with a
# non-zero bearing are what distinguish a bearing-relative result from one that only
# looks at the global motion direction.
_BEARING_RELATIVE_CASES = [
    ((1.0, 0.0), 0.0, 0.0, -180.0),
    ((0.0, 1.0), 0.0, 90.0, -90.0),
    ((-1.0, 0.0), 0.0, 180.0, 0.0),
    ((0.0, -1.0), 0.0, -90.0, 90.0),
    ((0.0, 1.0), 90.0, 0.0, -180.0),
    ((1.0, 0.0), 90.0, -90.0, 90.0),
    ((1.0, 0.0), -135.0, 135.0, -45.0),
    ((0.0, -1.0), 135.0, 135.0, -45.0),
]

_BEARING_RELATIVE_IDS = [
    "along-bearing",
    "90-off-bearing",
    "against-bearing",
    "270-off-bearing",
    "along-bearing-90",
    "90-off-bearing-90",
    "135-off-bearing-neg135",
    "135-off-bearing-135",
]


@pytest.mark.parametrize(
    ("delta", "bearing", "intended", "emitted"),
    _BEARING_RELATIVE_CASES,
    ids=_BEARING_RELATIVE_IDS,
)
def test_point_velocity_dirs_bearing_relative_convention(
    pose_est_v5, delta, bearing, intended, emitted
):
    """Pin the bearing-relative angle convention this feature emits.

    Every keypoint translates by ``delta`` each frame against a constant ``bearing``,
    which fixes the expected angle exactly. The non-zero-bearing cases are what make
    this a test of the bearing subtraction: the same ``delta`` is paired with different
    bearings and must give different answers, so an implementation that reported the
    global motion direction and ignored the bearing would fail.

    The expectations encode the 180 degree offset documented on
    :class:`PointVelocityDirs`: motion straight along the bearing reads ``-180`` rather
    than ``0``. ``intended`` is unused here and carried only to keep the case table
    readable alongside :func:`test_point_velocity_dirs_offset_is_a_constant_rotation`.
    Correcting the offset changes cached feature values, so it needs a
    ``FEATURE_VERSION`` bump - and rotating the ``emitted`` column by 180 degrees.
    """
    poses = _pose_with_constant_motion(pose_est_v5, delta, bearing)
    values = PointVelocityDirs(poses, 1.0).per_frame(0)

    for keypoint in PoseEstimation.KeypointIndex:
        direction = values[f"{keypoint.name} velocity direction"]
        assert direction == pytest.approx(emitted), keypoint.name

        # sine/cosine must stay consistent with the direction they are derived from
        assert values[f"{keypoint.name} velocity direction sine"] == pytest.approx(
            np.sin(np.deg2rad(emitted)), abs=1e-6
        )
        assert values[f"{keypoint.name} velocity direction cosine"] == pytest.approx(
            np.cos(np.deg2rad(emitted)), abs=1e-6
        )


@pytest.mark.parametrize(
    ("delta", "bearing", "intended", "emitted"),
    # the against-bearing case is excluded: its intended angle sits exactly on the
    # +/-180 wrap boundary, where 180 and -180 are the same angle
    [case for case in _BEARING_RELATIVE_CASES if abs(case[2]) != 180.0],
    ids=[
        name
        for name, case in zip(_BEARING_RELATIVE_IDS, _BEARING_RELATIVE_CASES, strict=True)
        if abs(case[2]) != 180.0
    ],
)
def test_point_velocity_dirs_offset_is_a_constant_rotation(
    pose_est_v5, delta, bearing, intended, emitted
):
    """Rotating the emitted angles by 180 degrees recovers the intended convention.

    This records that the gap against ``CentroidVelocityDir`` is one constant rotation
    rather than a sporadic discrepancy, so whoever fixes the wrap can confirm it closes
    the gap for every direction and bearing at once.
    """
    poses = _pose_with_constant_motion(pose_est_v5, delta, bearing)
    values = PointVelocityDirs(poses, 1.0).per_frame(0)

    for keypoint in PoseEstimation.KeypointIndex:
        # undo the offset, then wrap back into [-180, 180) the way
        # jabs.core.utils.geometry.signed_angle_degrees does
        rotated = ((values[f"{keypoint.name} velocity direction"] + 180.0) + 180.0) % 360.0 - 180.0
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
