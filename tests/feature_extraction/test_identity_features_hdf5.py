"""Integration tests for IdentityFeatures HDF5 caching.

These tests construct a real IdentityFeatures from a sample pose file, write the
HDF5 cache to a temporary directory, then read it back and verify the round-trip
produces identical feature arrays.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation as pose_est_module
from jabs.feature_extraction.features import IdentityFeatures

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


def _make_identity_features(pose_est, directory, *, force: bool) -> IdentityFeatures:
    """Construct an IdentityFeatures instance with consistent settings."""
    return IdentityFeatures(
        source_file=_SOURCE_FILE,
        identity=_IDENTITY,
        directory=directory,
        pose_est=pose_est,
        force=force,
        op_settings={},
    )


def test_per_frame_cache_round_trip(tmp_path, pose_est_v5) -> None:
    """Per-frame features loaded from cache must equal freshly computed values."""
    computed = _make_identity_features(pose_est_v5, tmp_path, force=True)
    cached = _make_identity_features(pose_est_v5, tmp_path, force=False)

    computed_flat = IdentityFeatures.merge_per_frame_features(computed.get_per_frame())
    cached_flat = IdentityFeatures.merge_per_frame_features(cached.get_per_frame())

    assert set(computed_flat) == set(cached_flat)
    for key in computed_flat:
        np.testing.assert_array_equal(cached_flat[key], computed_flat[key], err_msg=key)


def test_window_feature_cache_round_trip(tmp_path, pose_est_v5) -> None:
    """Window features loaded from cache must equal freshly computed values."""
    computed = _make_identity_features(pose_est_v5, tmp_path, force=True)
    computed_window = computed.get_window_features(_WINDOW_SIZE, force=True)

    cached = _make_identity_features(pose_est_v5, tmp_path, force=False)
    cached_window = cached.get_window_features(_WINDOW_SIZE)

    computed_flat = IdentityFeatures.merge_window_features(computed_window)
    cached_flat = IdentityFeatures.merge_window_features(cached_window)

    assert set(computed_flat) == set(cached_flat)
    for key in computed_flat:
        np.testing.assert_array_equal(cached_flat[key], computed_flat[key], err_msg=key)


def test_force_recompute_produces_consistent_results(tmp_path, pose_est_v5) -> None:
    """force=True recomputes correctly even when a valid cache already exists."""
    first = _make_identity_features(pose_est_v5, tmp_path, force=True)
    second = _make_identity_features(pose_est_v5, tmp_path, force=True)

    first_flat = IdentityFeatures.merge_per_frame_features(first.get_per_frame())
    second_flat = IdentityFeatures.merge_per_frame_features(second.get_per_frame())

    assert set(first_flat) == set(second_flat)
    for key in first_flat:
        np.testing.assert_array_equal(second_flat[key], first_flat[key], err_msg=key)
