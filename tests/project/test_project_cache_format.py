"""Tests for Project.cache_format property and related feature cache integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from jabs.core.constants import CACHE_FORMAT_KEY
from jabs.core.enums import CacheFormat
from jabs.project import Project


@pytest.fixture(autouse=True, scope="module")
def patch_session_tracker():
    """Suppress SessionTracker side effects."""
    with patch("jabs.project.session_tracker.SessionTracker.__del__", return_value=None):
        yield


def _make_project(tmp_path: Path) -> Project:
    """Create a minimal Project in tmp_path with no videos or pose files."""
    return Project(
        tmp_path,
        enable_video_check=False,
        enable_session_tracker=False,
        validate_project_dir=False,
    )


def _project_settings(tmp_path: Path) -> dict:
    """Read the settings dict from the project.json on disk."""
    project_file = tmp_path / "jabs" / "project.json"
    return json.loads(project_file.read_text()).get("settings", {})


# ---------------------------------------------------------------------------
# Default / migration behaviour
# ---------------------------------------------------------------------------


def test_new_project_defaults_to_parquet(tmp_path: Path) -> None:
    """A brand-new project gets cache_format=parquet written to project.json."""
    project = _make_project(tmp_path)

    assert project.cache_format == CacheFormat.PARQUET
    assert _project_settings(tmp_path).get(CACHE_FORMAT_KEY) == CacheFormat.PARQUET.value


def test_existing_project_without_cache_format_migrates_to_hdf5(tmp_path: Path) -> None:
    """A project.json that predates cache_format gets hdf5 written on open."""
    jabs_dir = tmp_path / "jabs"
    jabs_dir.mkdir(parents=True)
    project_file = jabs_dir / "project.json"
    # Write a project.json that has no cache_format key at all.
    project_file.write_text(json.dumps({"behavior": {}, "window_sizes": [5]}))

    project = _make_project(tmp_path)

    assert project.cache_format == CacheFormat.HDF5
    assert _project_settings(tmp_path).get(CACHE_FORMAT_KEY) == CacheFormat.HDF5.value


def test_existing_project_with_cache_format_parquet_is_preserved(tmp_path: Path) -> None:
    """Opening a project that already has cache_format=parquet preserves it."""
    jabs_dir = tmp_path / "jabs"
    jabs_dir.mkdir(parents=True)
    project_file = jabs_dir / "project.json"
    project_file.write_text(
        json.dumps({"behavior": {}, "settings": {CACHE_FORMAT_KEY: "parquet"}})
    )

    project = _make_project(tmp_path)

    assert project.cache_format == CacheFormat.PARQUET


# ---------------------------------------------------------------------------
# cache_format property — value handling
# ---------------------------------------------------------------------------


def test_cache_format_returns_enum_instance(tmp_path: Path) -> None:
    """cache_format returns a CacheFormat instance, not a bare string."""
    project = _make_project(tmp_path)

    result = project.cache_format
    assert isinstance(result, CacheFormat)


def test_cache_format_falls_back_to_hdf5_for_unknown_value(tmp_path: Path) -> None:
    """An unrecognised cache_format value in project.json falls back to HDF5."""
    jabs_dir = tmp_path / "jabs"
    jabs_dir.mkdir(parents=True)
    project_file = jabs_dir / "project.json"
    project_file.write_text(
        json.dumps({"behavior": {}, "settings": {CACHE_FORMAT_KEY: "unknown_format"}})
    )

    project = _make_project(tmp_path)

    assert project.cache_format == CacheFormat.HDF5


def test_clear_feature_cache_removes_hdf5_files(tmp_path: Path) -> None:
    """clear_feature_cache removes features.h5 from all identity dirs."""
    project = _make_project(tmp_path)

    identity_dir = project.feature_dir / "video_stem" / "0"
    identity_dir.mkdir(parents=True)
    (identity_dir / "features.h5").write_bytes(b"stub")

    project.clear_feature_cache()

    assert not (identity_dir / "features.h5").exists()


def test_clear_feature_cache_removes_parquet_files(tmp_path: Path) -> None:
    """clear_feature_cache removes Parquet cache files from all identity dirs."""
    project = _make_project(tmp_path)

    identity_dir = project.feature_dir / "video_stem" / "0"
    identity_dir.mkdir(parents=True)
    (identity_dir / "metadata.json").write_text("{}")
    (identity_dir / "per_frame.parquet").write_bytes(b"stub")
    (identity_dir / "window_5.parquet").write_bytes(b"stub")

    project.clear_feature_cache()

    assert not (identity_dir / "metadata.json").exists()
    assert not any(identity_dir.glob("*.parquet"))


def test_clear_feature_cache_preserves_directory_structure(tmp_path: Path) -> None:
    """clear_feature_cache removes files but leaves directories intact."""
    project = _make_project(tmp_path)

    identity_dir = project.feature_dir / "video_stem" / "0"
    identity_dir.mkdir(parents=True)
    (identity_dir / "features.h5").write_bytes(b"stub")

    project.clear_feature_cache()

    assert identity_dir.exists()
    assert (project.feature_dir / "video_stem").exists()


def test_clear_feature_cache_no_op_when_feature_dir_empty(tmp_path: Path) -> None:
    """clear_feature_cache does not raise when the features directory has no cache files."""
    project = _make_project(tmp_path)

    # feature_dir is created by Project.__init__; it just has no identity subdirs yet
    assert project.feature_dir.exists()
    project.clear_feature_cache()  # should not raise


def test_clear_feature_cache_handles_pose_hash_layout(tmp_path: Path) -> None:
    """clear_feature_cache removes cache files when pose-hash subdirectory is present.

    Hash layout: features/<video>/<pose_hash>/<identity>/
    """
    project = _make_project(tmp_path)

    pose_hash = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"  # 40-char hex
    identity_dir = project.feature_dir / "video_stem" / pose_hash / "0"
    identity_dir.mkdir(parents=True)
    (identity_dir / "metadata.json").write_text("{}")
    (identity_dir / "per_frame.parquet").write_bytes(b"stub")
    (identity_dir / "window_5.parquet").write_bytes(b"stub")

    project.clear_feature_cache()

    assert not (identity_dir / "metadata.json").exists()
    assert not any(identity_dir.glob("*.parquet"))
    assert identity_dir.exists()  # directories preserved
