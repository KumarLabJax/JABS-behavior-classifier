"""Tests for HDF5FeatureCacheReader and HDF5FeatureCacheWriter."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from jabs.core.exceptions import DistanceScaleException, FeatureVersionException
from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheReader, HDF5FeatureCacheWriter
from jabs.pose_estimation import PoseHashException

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_N_FRAMES = 50
_FEATURE_VERSION = 10
_POSE_HASH = "deadbeef"
_DISTANCE_SCALE: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metadata(**overrides) -> FeatureCacheMetadata:
    """Return a FeatureCacheMetadata with sensible defaults."""
    return FeatureCacheMetadata(
        feature_version=overrides.get("feature_version", _FEATURE_VERSION),
        identity=overrides.get("identity", 0),
        num_frames=overrides.get("num_frames", _N_FRAMES),
        pose_hash=overrides.get("pose_hash", _POSE_HASH),
        distance_scale_factor=overrides.get("distance_scale_factor", _DISTANCE_SCALE),
        avg_wall_length=overrides.get("avg_wall_length"),
    )


def _flat_features(rng: np.random.Generator, n: int = 5) -> dict[str, npt.NDArray[np.float64]]:
    """Return a flat feature dict with n synthetic columns."""
    return {f"mod feat_{i}": rng.standard_normal(_N_FRAMES) for i in range(n)}


def _per_frame_data(
    rng: np.random.Generator,
    *,
    with_aux: bool = False,
) -> PerFrameCacheData:
    """Return a PerFrameCacheData; optionally populate all optional fields."""
    frame_valid = rng.integers(0, 2, size=_N_FRAMES, dtype=np.uint8)
    features = _flat_features(rng)
    if not with_aux:
        return PerFrameCacheData(frame_valid=frame_valid, features=features)
    return PerFrameCacheData(
        frame_valid=frame_valid,
        features=features,
        closest_identities=rng.integers(0, 3, size=_N_FRAMES, dtype=np.int64),
        closest_fov_identities=rng.integers(0, 3, size=_N_FRAMES, dtype=np.int64),
        closest_corners=rng.standard_normal(_N_FRAMES),
        closest_lixit=rng.standard_normal(_N_FRAMES),
        wall_distances={
            "top": rng.standard_normal(_N_FRAMES),
            "bottom": rng.standard_normal(_N_FRAMES),
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def writer() -> HDF5FeatureCacheWriter:
    """Return an HDF5FeatureCacheWriter instance."""
    return HDF5FeatureCacheWriter()


@pytest.fixture
def reader() -> HDF5FeatureCacheReader:
    """Return an HDF5FeatureCacheReader configured with the default test values."""
    return HDF5FeatureCacheReader(_FEATURE_VERSION, _POSE_HASH, _DISTANCE_SCALE)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_hdf5_per_frame_write_read_round_trip(
    tmp_path, writer: HDF5FeatureCacheWriter, reader: HDF5FeatureCacheReader
) -> None:
    """Written per-frame arrays must be recovered identically on read."""
    rng = np.random.default_rng(0)
    identity_dir = tmp_path / "identity_0"
    data = _per_frame_data(rng)

    writer.write_per_frame(identity_dir, _metadata(), data)
    result = reader.read_per_frame(identity_dir)

    np.testing.assert_array_equal(result.frame_valid, data.frame_valid)
    assert set(result.features) == set(data.features)
    for key in data.features:
        np.testing.assert_array_equal(result.features[key], data.features[key], err_msg=key)
    assert result.closest_identities is None
    assert result.closest_corners is None
    assert result.closest_lixit is None
    assert result.wall_distances == {}


def test_hdf5_window_overwrite_does_not_raise(
    tmp_path, writer: HDF5FeatureCacheWriter, reader: HDF5FeatureCacheReader
) -> None:
    """Writing the same window size twice replaces the data without raising."""
    rng = np.random.default_rng(11)
    identity_dir = tmp_path / "identity_0"
    window_size = 5

    first_data = {f"mod op feat_{i}": rng.standard_normal(_N_FRAMES) for i in range(4)}
    second_data = {f"mod op feat_{i}": rng.standard_normal(_N_FRAMES) for i in range(4)}

    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))
    writer.write_window(identity_dir, _metadata(), window_size, first_data)
    writer.write_window(identity_dir, _metadata(), window_size, second_data)

    result = reader.read_window(identity_dir, window_size)
    for key in second_data:
        np.testing.assert_array_equal(result[key], second_data[key], err_msg=key)


def test_hdf5_window_write_read_round_trip(
    tmp_path, writer: HDF5FeatureCacheWriter, reader: HDF5FeatureCacheReader
) -> None:
    """Written window arrays must be recovered identically on read."""
    rng = np.random.default_rng(1)
    identity_dir = tmp_path / "identity_0"
    window_size = 5
    window_data = {f"mod op feat_{i}": rng.standard_normal(_N_FRAMES) for i in range(4)}

    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))
    writer.write_window(identity_dir, _metadata(), window_size, window_data)
    result = reader.read_window(identity_dir, window_size)

    assert set(result) == set(window_data)
    for key in window_data:
        np.testing.assert_array_equal(result[key], window_data[key], err_msg=key)


# ---------------------------------------------------------------------------
# Validation mismatch tests
# ---------------------------------------------------------------------------


def test_hdf5_version_mismatch(tmp_path, writer: HDF5FeatureCacheWriter) -> None:
    """FeatureVersionException raised when stored version differs from expected."""
    rng = np.random.default_rng(2)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    bad_reader = HDF5FeatureCacheReader(_FEATURE_VERSION + 1, _POSE_HASH, _DISTANCE_SCALE)
    with pytest.raises(FeatureVersionException):
        bad_reader.read_per_frame(identity_dir)


def test_hdf5_pose_hash_mismatch(tmp_path, writer: HDF5FeatureCacheWriter) -> None:
    """PoseHashException raised when stored hash differs from expected."""
    rng = np.random.default_rng(3)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    bad_reader = HDF5FeatureCacheReader(_FEATURE_VERSION, "wronghash", _DISTANCE_SCALE)
    with pytest.raises(PoseHashException):
        bad_reader.read_per_frame(identity_dir)


def test_hdf5_distance_scale_mismatch(tmp_path, writer: HDF5FeatureCacheWriter) -> None:
    """DistanceScaleException raised when stored scale differs from expected."""
    rng = np.random.default_rng(4)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    bad_reader = HDF5FeatureCacheReader(_FEATURE_VERSION, _POSE_HASH, 0.25)
    with pytest.raises(DistanceScaleException):
        bad_reader.read_per_frame(identity_dir)


def test_hdf5_window_size_not_cached(
    tmp_path, writer: HDF5FeatureCacheWriter, reader: HDF5FeatureCacheReader
) -> None:
    """AttributeError raised when the requested window size was never written."""
    rng = np.random.default_rng(5)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    with pytest.raises(AttributeError):
        reader.read_window(identity_dir, window_size=99)


# ---------------------------------------------------------------------------
# Social identity array pairing validation
# ---------------------------------------------------------------------------


def test_hdf5_closest_corners_without_avg_wall_length_raises(
    tmp_path, writer: HDF5FeatureCacheWriter
) -> None:
    """ValueError raised when closest_corners is present but avg_wall_length is missing."""
    rng = np.random.default_rng(10)
    identity_dir = tmp_path / "identity_0"
    data = PerFrameCacheData(
        frame_valid=rng.integers(0, 2, size=_N_FRAMES, dtype=np.uint8),
        features=_flat_features(rng),
        closest_corners=rng.standard_normal(_N_FRAMES),
    )
    with pytest.raises(ValueError, match="avg_wall_length"):
        writer.write_per_frame(identity_dir, _metadata(), data)


def test_hdf5_closest_identities_without_fov_raises(
    tmp_path, writer: HDF5FeatureCacheWriter
) -> None:
    """ValueError raised when closest_identities is provided without closest_fov_identities."""
    rng = np.random.default_rng(8)
    identity_dir = tmp_path / "identity_0"
    data = PerFrameCacheData(
        frame_valid=rng.integers(0, 2, size=_N_FRAMES, dtype=np.uint8),
        features=_flat_features(rng),
        closest_identities=rng.integers(0, 3, size=_N_FRAMES, dtype=np.int64),
        closest_fov_identities=None,
    )
    with pytest.raises(ValueError, match="closest_identities and closest_fov_identities"):
        writer.write_per_frame(identity_dir, _metadata(), data)


def test_hdf5_closest_fov_identities_without_identities_raises(
    tmp_path, writer: HDF5FeatureCacheWriter
) -> None:
    """ValueError raised when closest_fov_identities is provided without closest_identities."""
    rng = np.random.default_rng(9)
    identity_dir = tmp_path / "identity_0"
    data = PerFrameCacheData(
        frame_valid=rng.integers(0, 2, size=_N_FRAMES, dtype=np.uint8),
        features=_flat_features(rng),
        closest_identities=None,
        closest_fov_identities=rng.integers(0, 3, size=_N_FRAMES, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="closest_identities and closest_fov_identities"):
        writer.write_per_frame(identity_dir, _metadata(), data)


# ---------------------------------------------------------------------------
# Auxiliary field tests
# ---------------------------------------------------------------------------


def test_hdf5_auxiliary_optional_absent(
    tmp_path, writer: HDF5FeatureCacheWriter, reader: HDF5FeatureCacheReader
) -> None:
    """Reads correctly when no social or landmark auxiliary fields are present."""
    rng = np.random.default_rng(6)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng, with_aux=False))
    result = reader.read_per_frame(identity_dir)

    assert result.closest_identities is None
    assert result.closest_fov_identities is None
    assert result.closest_corners is None
    assert result.closest_lixit is None
    assert result.wall_distances == {}


def test_hdf5_auxiliary_present(
    tmp_path, writer: HDF5FeatureCacheWriter, reader: HDF5FeatureCacheReader
) -> None:
    """All optional auxiliary arrays round-trip correctly."""
    rng = np.random.default_rng(7)
    identity_dir = tmp_path / "identity_0"
    data = _per_frame_data(rng, with_aux=True)
    # avg_wall_length must be set when closest_corners is present.
    meta = _metadata(avg_wall_length=42.5)
    writer.write_per_frame(identity_dir, meta, data)
    result = reader.read_per_frame(identity_dir)

    np.testing.assert_array_equal(result.closest_identities, data.closest_identities)
    np.testing.assert_array_equal(result.closest_fov_identities, data.closest_fov_identities)
    np.testing.assert_array_equal(result.closest_corners, data.closest_corners)
    np.testing.assert_array_equal(result.closest_lixit, data.closest_lixit)
    assert set(result.wall_distances) == set(data.wall_distances)
    for direction in data.wall_distances:
        np.testing.assert_array_equal(
            result.wall_distances[direction],
            data.wall_distances[direction],
            err_msg=direction,
        )
