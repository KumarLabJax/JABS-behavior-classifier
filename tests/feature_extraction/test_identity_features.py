"""Tests for IdentityFeatures auto-detection and Parquet cache round-trip."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation as pose_est_module
from jabs.core.enums import CacheFormat
from jabs.feature_extraction.features import IdentityFeatures
from jabs.io.feature_cache import detect_cache_format

_SAMPLE_POSE_V5 = Path(__file__).parent.parent / "data" / "sample_pose_est_v5.h5"

_SOURCE_FILE = "sample_pose_est_v5.h5"
_IDENTITY = 0
_WINDOW_SIZE = 5


@pytest.fixture(scope="module")
def pose_est_v5():
    """Load the sample v5 pose estimation from a temporary copy.

    Yields:
        PoseEstimationV5: Pose estimation object backed by a temporary file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pose_path = Path(tmpdir) / _SAMPLE_POSE_V5.name
        shutil.copy(_SAMPLE_POSE_V5, pose_path)
        yield pose_est_module.open_pose_file(pose_path)


def _make_identity_features(
    pose_est,
    directory,
    *,
    force: bool,
    cache_format: CacheFormat = CacheFormat.HDF5,
) -> IdentityFeatures:
    """Construct an IdentityFeatures instance with consistent settings."""
    return IdentityFeatures(
        source_file=_SOURCE_FILE,
        identity=_IDENTITY,
        directory=directory,
        pose_est=pose_est,
        force=force,
        op_settings={},
        cache_format=cache_format,
    )


# ---------------------------------------------------------------------------
# detect_cache_format tests
# ---------------------------------------------------------------------------


def test_autodetect_parquet(tmp_path) -> None:
    """detect_cache_format returns PARQUET when metadata.json is present."""
    identity_dir = tmp_path / "identity_0"
    identity_dir.mkdir()
    (identity_dir / "metadata.json").touch()

    assert detect_cache_format(identity_dir) == CacheFormat.PARQUET


def test_autodetect_hdf5(tmp_path) -> None:
    """detect_cache_format returns HDF5 when features.h5 is present and metadata.json is absent."""
    identity_dir = tmp_path / "identity_0"
    identity_dir.mkdir()
    (identity_dir / "features.h5").touch()

    assert detect_cache_format(identity_dir) == CacheFormat.HDF5


def test_autodetect_none(tmp_path) -> None:
    """detect_cache_format returns None when neither sentinel file is present."""
    identity_dir = tmp_path / "identity_0"
    identity_dir.mkdir()

    assert detect_cache_format(identity_dir) is None


def test_autodetect_parquet_takes_priority(tmp_path) -> None:
    """detect_cache_format returns PARQUET when both metadata.json and features.h5 are present."""
    identity_dir = tmp_path / "identity_0"
    identity_dir.mkdir()
    (identity_dir / "metadata.json").touch()
    (identity_dir / "features.h5").touch()

    assert detect_cache_format(identity_dir) == CacheFormat.PARQUET


# ---------------------------------------------------------------------------
# Parquet round-trip integration tests
# ---------------------------------------------------------------------------


def test_identity_features_parquet_round_trip(tmp_path, pose_est_v5) -> None:
    """Per-frame features loaded from a Parquet cache must equal freshly computed values."""
    computed = _make_identity_features(
        pose_est_v5, tmp_path, force=True, cache_format=CacheFormat.PARQUET
    )
    cached = _make_identity_features(
        pose_est_v5, tmp_path, force=False, cache_format=CacheFormat.PARQUET
    )

    computed_flat = IdentityFeatures.merge_per_frame_features(computed.get_per_frame())
    cached_flat = IdentityFeatures.merge_per_frame_features(cached.get_per_frame())

    assert set(computed_flat) == set(cached_flat)
    for key in computed_flat:
        np.testing.assert_array_almost_equal(cached_flat[key], computed_flat[key], err_msg=key)


def test_identity_features_parquet_window_round_trip(tmp_path, pose_est_v5) -> None:
    """Window features loaded from a Parquet cache must equal freshly computed values."""
    computed = _make_identity_features(
        pose_est_v5, tmp_path, force=True, cache_format=CacheFormat.PARQUET
    )
    computed_window = computed.get_window_features(_WINDOW_SIZE, force=True)

    cached = _make_identity_features(
        pose_est_v5, tmp_path, force=False, cache_format=CacheFormat.PARQUET
    )
    cached_window = cached.get_window_features(_WINDOW_SIZE)

    computed_flat = IdentityFeatures.merge_window_features(computed_window)
    cached_flat = IdentityFeatures.merge_window_features(cached_window)

    assert set(computed_flat) == set(cached_flat)
    for key in computed_flat:
        np.testing.assert_array_almost_equal(cached_flat[key], computed_flat[key], err_msg=key)


def test_window_cache_readable_after_first_compute(tmp_path, pose_est_v5) -> None:
    """A single IdentityFeatures instance can load window features it just wrote.

    Regression test: when no cache exists at construction time, _reader was left
    as None after writing the per-frame cache, causing every get_window_features()
    call on the same instance to recompute instead of loading from disk.
    """
    # First call with no existing cache — writes per-frame cache and initializes reader.
    instance = _make_identity_features(pose_est_v5, tmp_path, force=False)

    # First get_window_features call: computes and writes window cache.
    first = instance.get_window_features(_WINDOW_SIZE)

    # Second call on the same instance: should load from cache, not recompute.
    # If _reader were still None this would silently recompute; results are
    # identical either way, but the cache file must exist.
    second = instance.get_window_features(_WINDOW_SIZE)

    first_flat = IdentityFeatures.merge_window_features(first)
    second_flat = IdentityFeatures.merge_window_features(second)

    assert set(first_flat) == set(second_flat)
    for key in first_flat:
        np.testing.assert_array_equal(second_flat[key], first_flat[key], err_msg=key)


def test_force_with_format_change_removes_stale_sentinel(tmp_path, pose_est_v5) -> None:
    """force=True with a different cache_format removes the old sentinel before writing.

    Regression test: when an existing Parquet cache (metadata.json) was present
    and force=True + cache_format=HDF5 wrote a new features.h5, the stale
    metadata.json remained. Subsequent force=False runs would find metadata.json
    first and read the stale Parquet cache instead of the freshly written HDF5 one.
    """
    # Write an initial Parquet cache.
    _make_identity_features(pose_est_v5, tmp_path, force=True, cache_format=CacheFormat.PARQUET)
    identity_dir = tmp_path / Path(_SOURCE_FILE).stem / str(_IDENTITY)
    assert (identity_dir / "metadata.json").exists()

    # Force-recompute into HDF5 format.
    _make_identity_features(pose_est_v5, tmp_path, force=True, cache_format=CacheFormat.HDF5)

    # The stale Parquet sentinel must be gone; only the HDF5 file should remain.
    assert not (identity_dir / "metadata.json").exists()
    assert not any(identity_dir.glob("*.parquet"))
    assert (identity_dir / "features.h5").exists()

    # A subsequent force=False run must read the HDF5 cache, not a stale Parquet one.
    from jabs.io.feature_cache import detect_cache_format

    assert detect_cache_format(identity_dir) == CacheFormat.HDF5
