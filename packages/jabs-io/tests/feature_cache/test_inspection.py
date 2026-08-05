"""Tests for read-only feature cache inspection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jabs.core.enums import CacheFormat
from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.io.feature_cache import inspect_identity_cache
from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheWriter
from jabs.io.feature_cache.parquet import ParquetFeatureCacheWriter

_N_FRAMES = 20
_FEATURE_VERSION = 11
_POSE_HASH = "abc123"


def _metadata(**overrides) -> FeatureCacheMetadata:
    """Return cache metadata with test defaults."""
    return FeatureCacheMetadata(
        feature_version=overrides.get("feature_version", _FEATURE_VERSION),
        identity=overrides.get("identity", 2),
        num_frames=overrides.get("num_frames", _N_FRAMES),
        pose_hash=overrides.get("pose_hash", _POSE_HASH),
        distance_scale_factor=overrides.get("distance_scale_factor"),
    )


def _per_frame_data() -> PerFrameCacheData:
    """Return minimal per-frame cache data."""
    rng = np.random.default_rng(seed=7)
    return PerFrameCacheData(
        frame_valid=np.ones(_N_FRAMES, dtype=np.uint8),
        features={"mod feat": rng.standard_normal(_N_FRAMES)},
    )


def _window_data() -> dict[str, np.ndarray]:
    """Return minimal window feature data."""
    rng = np.random.default_rng(seed=8)
    return {"mod mean feat": rng.standard_normal(_N_FRAMES)}


def _write_cache(
    identity_dir: Path,
    cache_format: CacheFormat,
    window_sizes: tuple[int, ...] = (),
    **metadata_overrides,
) -> None:
    """Write a cache in the requested format with the given window sizes."""
    writer = (
        ParquetFeatureCacheWriter()
        if cache_format == CacheFormat.PARQUET
        else HDF5FeatureCacheWriter()
    )
    metadata = _metadata(**metadata_overrides)
    writer.write_per_frame(identity_dir, metadata, _per_frame_data())
    for size in window_sizes:
        writer.write_window(identity_dir, metadata, size, _window_data())


@pytest.mark.parametrize(
    "cache_format", [CacheFormat.HDF5, CacheFormat.PARQUET], ids=["hdf5", "parquet"]
)
def test_inspect_reports_metadata_and_window_sizes(tmp_path, cache_format):
    """Inspection reports the metadata and window sizes written to the cache."""
    identity_dir = tmp_path / "2"
    _write_cache(identity_dir, cache_format, window_sizes=(5, 10))

    info = inspect_identity_cache(identity_dir)

    assert info is not None
    assert info.directory == identity_dir
    assert info.cache_format == cache_format
    assert info.identity == 2
    assert info.feature_version == _FEATURE_VERSION
    assert info.pose_hash == _POSE_HASH
    assert info.num_frames == _N_FRAMES
    assert info.window_sizes == frozenset({5, 10})
    assert info.per_frame_present is True
    assert info.distance_scale_factor is None
    assert info.size_bytes > 0


@pytest.mark.parametrize(
    "cache_format", [CacheFormat.HDF5, CacheFormat.PARQUET], ids=["hdf5", "parquet"]
)
def test_inspect_reports_no_window_sizes_when_none_cached(tmp_path, cache_format):
    """A cache with only per-frame features reports an empty window size set."""
    identity_dir = tmp_path / "0"
    _write_cache(identity_dir, cache_format, identity=0)

    info = inspect_identity_cache(identity_dir)

    assert info is not None
    assert info.window_sizes == frozenset()


@pytest.mark.parametrize(
    "cache_format", [CacheFormat.HDF5, CacheFormat.PARQUET], ids=["hdf5", "parquet"]
)
def test_inspect_reports_distance_scale_factor(tmp_path, cache_format):
    """The distance scale factor a cache was computed with is reported."""
    identity_dir = tmp_path / "1"
    _write_cache(identity_dir, cache_format, identity=1, distance_scale_factor=0.25)

    info = inspect_identity_cache(identity_dir)

    assert info is not None
    assert info.distance_scale_factor == pytest.approx(0.25)


def test_inspect_missing_cache_returns_none(tmp_path):
    """A directory with no cache files yields None."""
    empty = tmp_path / "0"
    empty.mkdir()

    assert inspect_identity_cache(empty) is None


def test_inspect_nonexistent_directory_returns_none(tmp_path):
    """A directory that does not exist yields None rather than raising."""
    assert inspect_identity_cache(tmp_path / "does-not-exist") is None


def test_inspect_unreadable_hdf5_returns_none(tmp_path):
    """A features.h5 that is not a valid HDF5 file yields None."""
    identity_dir = tmp_path / "0"
    identity_dir.mkdir()
    (identity_dir / "features.h5").write_text("not hdf5")

    assert inspect_identity_cache(identity_dir) is None


def test_inspect_malformed_metadata_json_returns_none(tmp_path):
    """A metadata.json that is not valid JSON yields None."""
    identity_dir = tmp_path / "0"
    identity_dir.mkdir()
    (identity_dir / "metadata.json").write_text("{not json")

    assert inspect_identity_cache(identity_dir) is None


def test_inspect_metadata_json_missing_field_returns_none(tmp_path):
    """A metadata.json missing a required field yields None."""
    identity_dir = tmp_path / "0"
    identity_dir.mkdir()
    (identity_dir / "metadata.json").write_text(json.dumps({"identity": 0}))

    assert inspect_identity_cache(identity_dir) is None


def test_inspect_omits_registered_window_size_with_missing_file(tmp_path):
    """A registered Parquet window size whose file is gone is not reported."""
    identity_dir = tmp_path / "0"
    _write_cache(identity_dir, CacheFormat.PARQUET, window_sizes=(5, 10), identity=0)
    (identity_dir / "window_10.parquet").unlink()

    info = inspect_identity_cache(identity_dir)

    assert info is not None
    assert info.window_sizes == frozenset({5})


def test_inspect_reports_missing_parquet_per_frame_features(tmp_path):
    """A Parquet cache without per_frame.parquet reports per_frame_present False."""
    identity_dir = tmp_path / "0"
    _write_cache(identity_dir, CacheFormat.PARQUET, identity=0)
    (identity_dir / "per_frame.parquet").unlink()

    info = inspect_identity_cache(identity_dir)

    assert info is not None
    assert info.per_frame_present is False


def test_inspect_prefers_parquet_when_both_formats_present(tmp_path):
    """metadata.json takes precedence, matching detect_cache_format()."""
    identity_dir = tmp_path / "0"
    _write_cache(identity_dir, CacheFormat.HDF5, identity=0)
    _write_cache(identity_dir, CacheFormat.PARQUET, identity=0)

    info = inspect_identity_cache(identity_dir)

    assert info is not None
    assert info.cache_format == CacheFormat.PARQUET


def test_inspect_survives_an_unreadable_cache_directory(tmp_path, monkeypatch):
    """A directory listing failure while sizing the cache does not fail inspection.

    Size is one field of many; a cache directory that becomes unreadable (or is
    removed) mid-scan must not abort a project-wide scan.
    """
    identity_dir = tmp_path / "0"
    _write_cache(identity_dir, CacheFormat.HDF5, window_sizes=(5,), identity=0)

    def _deny_listing(_self):
        raise PermissionError("cache directory is not readable")

    monkeypatch.setattr(Path, "iterdir", _deny_listing)

    info = inspect_identity_cache(identity_dir)

    assert info is not None
    assert info.window_sizes == frozenset({5})
    assert info.size_bytes == 0
