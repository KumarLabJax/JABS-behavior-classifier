"""Tests for ParquetFeatureCacheReader and ParquetFeatureCacheWriter."""

from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
import pytest

from jabs.core.exceptions import DistanceScaleException, FeatureVersionException
from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.io.feature_cache import detect_cache_format
from jabs.io.feature_cache.parquet import (
    PARQUET_FORMAT_VERSION,
    ParquetFeatureCacheReader,
    ParquetFeatureCacheWriter,
)
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
def writer() -> ParquetFeatureCacheWriter:
    """Return a ParquetFeatureCacheWriter instance."""
    return ParquetFeatureCacheWriter()


@pytest.fixture
def reader() -> ParquetFeatureCacheReader:
    """Return a ParquetFeatureCacheReader configured with the default test values."""
    return ParquetFeatureCacheReader(_FEATURE_VERSION, _POSE_HASH, _DISTANCE_SCALE)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_parquet_per_frame_round_trip(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
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
        np.testing.assert_array_almost_equal(result.features[key], data.features[key], err_msg=key)
    assert result.closest_identities is None
    assert result.closest_corners is None
    assert result.closest_lixit is None
    assert result.wall_distances == {}


def test_parquet_window_round_trip(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
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
        np.testing.assert_array_almost_equal(result[key], window_data[key], err_msg=key)


def test_parquet_multiple_window_sizes(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
) -> None:
    """Both window sizes are readable and cached_window_sizes is updated correctly."""
    rng = np.random.default_rng(2)
    identity_dir = tmp_path / "identity_0"
    data_5 = {f"mod op feat_{i}": rng.standard_normal(_N_FRAMES) for i in range(3)}
    data_10 = {f"mod op feat_{i}": rng.standard_normal(_N_FRAMES) for i in range(3)}

    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))
    writer.write_window(identity_dir, _metadata(), 5, data_5)
    writer.write_window(identity_dir, _metadata(), 10, data_10)

    metadata = reader.read_metadata(identity_dir)
    assert metadata.cached_window_sizes == frozenset({5, 10})

    result_5 = reader.read_window(identity_dir, 5)
    result_10 = reader.read_window(identity_dir, 10)
    for key in data_5:
        np.testing.assert_array_almost_equal(result_5[key], data_5[key], err_msg=key)
    for key in data_10:
        np.testing.assert_array_almost_equal(result_10[key], data_10[key], err_msg=key)


# ---------------------------------------------------------------------------
# Sentinel / write-ordering test
# ---------------------------------------------------------------------------


def test_parquet_metadata_sentinel(tmp_path, writer: ParquetFeatureCacheWriter) -> None:
    """detect_cache_format returns None when only per_frame.parquet exists (no metadata.json).

    This simulates a crash between the Parquet write and the metadata.json write,
    ensuring the incomplete cache is safely ignored.
    """
    identity_dir = tmp_path / "identity_0"
    identity_dir.mkdir()
    # Simulate an incomplete write: per_frame.parquet present, metadata.json absent.
    (identity_dir / "per_frame.parquet").touch()

    assert detect_cache_format(identity_dir) is None


# ---------------------------------------------------------------------------
# Validation mismatch tests
# ---------------------------------------------------------------------------


def test_parquet_version_mismatch_feature(tmp_path, writer: ParquetFeatureCacheWriter) -> None:
    """FeatureVersionException raised when stored feature_version differs from expected."""
    rng = np.random.default_rng(3)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    bad_reader = ParquetFeatureCacheReader(_FEATURE_VERSION + 1, _POSE_HASH, _DISTANCE_SCALE)
    with pytest.raises(FeatureVersionException):
        bad_reader.read_per_frame(identity_dir)


def test_parquet_version_mismatch_format(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
) -> None:
    """FeatureVersionException raised when stored format_version differs from PARQUET_FORMAT_VERSION."""
    rng = np.random.default_rng(4)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    # Overwrite metadata.json with a wrong format_version.
    metadata_path = identity_dir / "metadata.json"
    raw = json.loads(metadata_path.read_text())
    raw["format_version"] = PARQUET_FORMAT_VERSION + 99
    metadata_path.write_text(json.dumps(raw))

    with pytest.raises(FeatureVersionException):
        reader.read_per_frame(identity_dir)


def test_parquet_pose_hash_mismatch(tmp_path, writer: ParquetFeatureCacheWriter) -> None:
    """PoseHashException raised when stored pose_hash differs from expected."""
    rng = np.random.default_rng(5)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    bad_reader = ParquetFeatureCacheReader(_FEATURE_VERSION, "wronghash", _DISTANCE_SCALE)
    with pytest.raises(PoseHashException):
        bad_reader.read_per_frame(identity_dir)


def test_parquet_distance_scale_mismatch(tmp_path, writer: ParquetFeatureCacheWriter) -> None:
    """DistanceScaleException raised when stored distance_scale_factor differs from expected."""
    rng = np.random.default_rng(6)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    bad_reader = ParquetFeatureCacheReader(_FEATURE_VERSION, _POSE_HASH, 0.25)
    with pytest.raises(DistanceScaleException):
        bad_reader.read_per_frame(identity_dir)


def test_parquet_window_size_not_cached(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
) -> None:
    """AttributeError raised when the requested window size was never written."""
    rng = np.random.default_rng(7)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))

    with pytest.raises(AttributeError):
        reader.read_window(identity_dir, window_size=99)


# ---------------------------------------------------------------------------
# Error recovery tests
# ---------------------------------------------------------------------------


def test_parquet_partial_loss_per_frame(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
) -> None:
    """OSError raised when per_frame.parquet is missing; all cache files are deleted."""
    rng = np.random.default_rng(8)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))
    writer.write_window(
        identity_dir,
        _metadata(),
        5,
        {f"mod op f_{i}": rng.standard_normal(_N_FRAMES) for i in range(3)},
    )

    (identity_dir / "per_frame.parquet").unlink()

    with pytest.raises(OSError):
        reader.read_per_frame(identity_dir)

    # All cache files must be deleted so detect_cache_format returns None.
    assert not (identity_dir / "metadata.json").exists()
    assert not any(identity_dir.glob("*.parquet"))


def test_parquet_partial_loss_window(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
) -> None:
    """AttributeError raised when a window file is missing; only that size is removed from metadata."""
    rng = np.random.default_rng(9)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))
    writer.write_window(
        identity_dir,
        _metadata(),
        5,
        {f"mod op f_{i}": rng.standard_normal(_N_FRAMES) for i in range(3)},
    )
    writer.write_window(
        identity_dir,
        _metadata(),
        10,
        {f"mod op f_{i}": rng.standard_normal(_N_FRAMES) for i in range(3)},
    )

    (identity_dir / "window_5.parquet").unlink()

    with pytest.raises(AttributeError):
        reader.read_window(identity_dir, 5)

    # Only size 5 must be removed; size 10 must remain.
    metadata = reader.read_metadata(identity_dir)
    assert 5 not in metadata.cached_window_sizes
    assert 10 in metadata.cached_window_sizes


# ---------------------------------------------------------------------------
# Auxiliary field tests
# ---------------------------------------------------------------------------


def test_parquet_auxiliary_optional_absent(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
) -> None:
    """Reads correctly when no social or landmark auxiliary fields are present."""
    rng = np.random.default_rng(10)
    identity_dir = tmp_path / "identity_0"
    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng, with_aux=False))
    result = reader.read_per_frame(identity_dir)

    assert result.closest_identities is None
    assert result.closest_fov_identities is None
    assert result.closest_corners is None
    assert result.closest_lixit is None
    assert result.wall_distances == {}


def test_parquet_auxiliary_present(
    tmp_path, writer: ParquetFeatureCacheWriter, reader: ParquetFeatureCacheReader
) -> None:
    """All optional auxiliary arrays round-trip correctly."""
    rng = np.random.default_rng(11)
    identity_dir = tmp_path / "identity_0"
    data = _per_frame_data(rng, with_aux=True)
    meta = _metadata(avg_wall_length=42.5)
    writer.write_per_frame(identity_dir, meta, data)
    result = reader.read_per_frame(identity_dir)

    np.testing.assert_array_equal(result.closest_identities, data.closest_identities)
    np.testing.assert_array_equal(result.closest_fov_identities, data.closest_fov_identities)
    np.testing.assert_array_almost_equal(result.closest_corners, data.closest_corners)
    np.testing.assert_array_almost_equal(result.closest_lixit, data.closest_lixit)
    assert set(result.wall_distances) == set(data.wall_distances)
    for direction in data.wall_distances:
        np.testing.assert_array_almost_equal(
            result.wall_distances[direction],
            data.wall_distances[direction],
            err_msg=direction,
        )


# ---------------------------------------------------------------------------
# Compression test
# ---------------------------------------------------------------------------


def test_parquet_lz4_compression(tmp_path, writer: ParquetFeatureCacheWriter) -> None:
    """per_frame.parquet and window Parquet files must use LZ4 compression."""
    rng = np.random.default_rng(12)
    identity_dir = tmp_path / "identity_0"
    window_data = {f"mod op feat_{i}": rng.standard_normal(_N_FRAMES) for i in range(3)}

    writer.write_per_frame(identity_dir, _metadata(), _per_frame_data(rng))
    writer.write_window(identity_dir, _metadata(), 5, window_data)

    for parquet_path in [identity_dir / "per_frame.parquet", identity_dir / "window_5.parquet"]:
        file_metadata = pq.read_metadata(parquet_path)
        for rg_idx in range(file_metadata.num_row_groups):
            row_group = file_metadata.row_group(rg_idx)
            for col_idx in range(row_group.num_columns):
                compression = row_group.column(col_idx).compression
                assert compression.lower() == "lz4", (
                    f"{parquet_path.name} column {col_idx} uses {compression!r}, expected LZ4"
                )
