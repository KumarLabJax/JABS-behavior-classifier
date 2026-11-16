"""Unit tests for the PointVelocityDirs feature class."""

import typing
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pytest
import shapely

from jabs.feature_extraction.base_features import CentroidVelocityDir, PointVelocityDirs
from jabs.pose_estimation import PoseEstimation


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


def test_point_velocity_dirs_sine_cosine_match_direction(pose_est_v5) -> None:
    """Test that each keypoint's sine and cosine columns are computed from its direction."""
    velocity_dirs_feature = PointVelocityDirs(pose_est_v5, pose_est_v5.cm_per_pixel)

    for identity in range(pose_est_v5.num_identities):
        values = velocity_dirs_feature.per_frame(identity)

        direction_names = [name for name in values if not name.endswith((" sine", " cosine"))]
        assert direction_names
        for name in direction_names:
            direction = values[name]
            assert np.isfinite(direction).any(), f"{name} has no valid frames to compare"
            np.testing.assert_allclose(
                values[f"{name} sine"], np.sin(np.deg2rad(direction)), atol=1e-6, err_msg=name
            )
            np.testing.assert_allclose(
                values[f"{name} cosine"], np.cos(np.deg2rad(direction)), atol=1e-6, err_msg=name
            )


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

    def get_identity_convex_hulls(self, identity: int) -> list[shapely.Polygon]:
        """Return the per-frame body hull of one identity (used by CentroidVelocityDir)."""
        ...

    def identity_mask(self, identity: int) -> npt.NDArray[np.uint8]:
        """Return 1 for frames the identity is present (used by CentroidVelocityDir)."""
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
    # a unit square that moves with the keypoints, so its centroid has the same motion
    hulls = [
        shapely.box(x, y, x + 1, y + 1)
        for x, y in zip(poses[:, 0, 0], poses[:, 0, 1], strict=True)
    ]

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

        def get_identity_convex_hulls(self, identity: int) -> list[shapely.Polygon]:
            """Return a unit square moving with the keypoints, for any identity."""
            return hulls

        def identity_mask(self, identity: int) -> npt.NDArray[np.uint8]:
            """Report the identity present in every frame."""
            return np.ones(num_frames, dtype=np.uint8)

    return typing.cast("_PoseLike", _PoseWithConstantMotion())


# (delta, bearing, expected): `expected` is the bearing-relative angle, in [-180, 180).
# The cases with a non-zero bearing are what distinguish a bearing-relative result from
# one that only looks at the global motion direction.
_BEARING_RELATIVE_CASES = [
    ((1.0, 0.0), 0.0, 0.0),
    ((0.0, 1.0), 0.0, 90.0),
    # directly against the bearing sits on the wrap boundary, which is -180, not 180
    ((-1.0, 0.0), 0.0, -180.0),
    ((0.0, -1.0), 0.0, -90.0),
    ((0.0, 1.0), 90.0, 0.0),
    ((1.0, 0.0), 90.0, -90.0),
    ((1.0, 0.0), -135.0, 135.0),
    ((0.0, -1.0), 135.0, 135.0),
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
    ("delta", "bearing", "expected"),
    _BEARING_RELATIVE_CASES,
    ids=_BEARING_RELATIVE_IDS,
)
def test_point_velocity_dirs_bearing_relative_convention(pose_est_v5, delta, bearing, expected):
    """Pin the bearing-relative angle convention this feature emits.

    Every keypoint translates by ``delta`` each frame against a constant ``bearing``,
    which fixes the expected angle exactly. The non-zero-bearing cases are what make
    this a test of the bearing subtraction: the same ``delta`` is paired with different
    bearings and must give different answers, so an implementation that reported the
    global motion direction and ignored the bearing would fail.

    Motion straight along the bearing reads ``0``, the same convention as
    ``CentroidVelocityDir``. Before ``FEATURE_VERSION`` 19 it read ``-180``.
    """
    poses = _pose_with_constant_motion(pose_est_v5, delta, bearing)
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
    ("delta", "bearing", "expected"),
    _BEARING_RELATIVE_CASES,
    ids=_BEARING_RELATIVE_IDS,
)
def test_point_velocity_dirs_matches_centroid_velocity_dir(
    pose_est_v5, delta, bearing, expected
) -> None:
    """Keypoint and centroid velocity directions use the same bearing convention.

    When every keypoint and the body hull move together, each keypoint's direction must
    equal the centroid's. This keeps the two features from drifting apart again, which
    is how the pre-``FEATURE_VERSION`` 19 offset arose.
    """
    poses = _pose_with_constant_motion(pose_est_v5, delta, bearing)
    point_values = PointVelocityDirs(poses, 1.0).per_frame(0)
    centroid = CentroidVelocityDir(poses, 1.0).per_frame(0)["centroid_velocity_dir"]

    assert centroid == pytest.approx(expected)
    for keypoint in PoseEstimation.KeypointIndex:
        np.testing.assert_allclose(
            point_values[f"{keypoint.name} velocity direction"],
            centroid,
            atol=1e-4,
            err_msg=keypoint.name,
        )


def test_point_velocity_dirs_name_and_circular_flag() -> None:
    """Test that the feature name is set correctly and circular statistics are enabled."""
    assert PointVelocityDirs.name() == "point_velocity_dirs"
    assert PointVelocityDirs._use_circular is True
