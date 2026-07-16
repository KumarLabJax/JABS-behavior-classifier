"""Tests for the per-video pose attribute cache (KLAUS-506)."""

import json
import os
import shutil
from pathlib import Path

import pytest

from jabs.project import Project, pose_attribute_cache
from jabs.project.project import _pose_cache_entry_matches

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_POSE = DATA_DIR / "sample_pose_est_v3.h5"
CACHE_RELPATH = Path("jabs") / "cache" / "pose_attribute_cache.json"


def _make_project(project_dir: Path, video_names: list[str]) -> Path:
    """Create a minimal project dir: stub ``.avi`` videos + copied v3 pose files."""
    project_dir.mkdir(parents=True, exist_ok=True)
    for name in video_names:
        (project_dir / name).touch()
        base = Path(name).with_suffix("").name
        shutil.copy(SAMPLE_POSE, project_dir / f"{base}_pose_est_v3.h5")
    return project_dir


def _open(project_dir: Path) -> Project:
    """Open a project on the deferred (cached) scan path."""
    return Project(project_dir, enable_video_check=False, enable_session_tracker=False)


def _spy_scan(monkeypatch) -> list[str]:
    """Spy on ``scan_video_metadata`` as Project uses it; returns scanned videos."""
    import jabs.project.project as project_mod

    real = project_mod.scan_video_metadata
    calls: list[str] = []

    def spy(job):
        calls.append(job["video"])
        return real(job)

    monkeypatch.setattr(project_mod, "scan_video_metadata", spy)
    return calls


def _bump_mtime(path: Path) -> None:
    """Advance a file's modification time by one second (changes its token)."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


# --- pose_attribute_cache module unit tests --------------------------------


def test_pose_token_changes_with_mtime(tmp_path):
    """The stat token changes when the file's modification time changes."""
    f = tmp_path / "x.h5"
    f.write_bytes(b"abc")
    before = pose_attribute_cache.pose_token(f)
    _bump_mtime(f)
    assert pose_attribute_cache.pose_token(f) != before


def test_load_none_returns_empty():
    """Caching disabled (None path) loads an empty map."""
    assert pose_attribute_cache.load(None) == {}


def test_load_missing_returns_empty(tmp_path):
    """A missing cache file loads an empty map."""
    assert pose_attribute_cache.load(tmp_path / "nope.json") == {}


def test_load_corrupt_returns_empty(tmp_path):
    """An unparseable cache file is treated as empty (triggers rescan)."""
    p = tmp_path / "c.json"
    p.write_text("{ not valid json")
    assert pose_attribute_cache.load(p) == {}


def test_load_schema_mismatch_returns_empty(tmp_path):
    """A cache written by a different schema version is ignored."""
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {"schema_version": pose_attribute_cache.SCHEMA_VERSION + 1, "videos": {"a.avi": {}}}
        )
    )
    assert pose_attribute_cache.load(p) == {}


def test_save_load_roundtrip(tmp_path):
    """A saved map is returned verbatim by load."""
    p = tmp_path / "c.json"
    videos = {"v.avi": {"token": "1:2", "pose_file": "v_pose_est_v3.h5", "hdf5_frame_count": 10}}
    pose_attribute_cache.save(p, videos)
    assert pose_attribute_cache.load(p) == videos


def test_save_none_is_noop():
    """Saving with a None path does nothing and does not raise."""
    pose_attribute_cache.save(None, {"v.avi": {}})


# --- _pose_cache_entry_matches unit tests ----------------------------------


def _valid_entry() -> dict:
    """Return a well-formed cache entry for token ``"1:2"`` / pose ``"v.h5"``."""
    return {
        "token": "1:2",
        "pose_file": "v.h5",
        "hdf5_frame_count": 10,
        "identity_count": 2,
        "static_objects": ["lixit", "food_hopper"],
        "lixit_keypoints": 3,
        "has_cm_per_pixel": True,
    }


def test_matches_well_formed_entry():
    """A complete, correctly-typed entry with matching token/pose is a hit."""
    assert _pose_cache_entry_matches(_valid_entry(), "1:2", "v.h5")


def test_matches_empty_static_objects():
    """An empty static_objects list is valid (no static objects in the pose file)."""
    entry = _valid_entry()
    entry["static_objects"] = []
    assert _pose_cache_entry_matches(entry, "1:2", "v.h5")


@pytest.mark.parametrize(
    ("token", "pose_file"),
    [("9:9", "v.h5"), ("1:2", "other.h5")],
    ids=["stale-token", "different-pose-file"],
)
def test_no_match_on_token_or_pose_mismatch(token, pose_file):
    """A mismatched token or pose filename is a miss."""
    assert not _pose_cache_entry_matches(_valid_entry(), token, pose_file)


def test_no_match_when_not_dict():
    """A non-dict entry (e.g. None for a missing video) is a miss."""
    assert not _pose_cache_entry_matches(None, "1:2", "v.h5")


def test_no_match_on_missing_field():
    """An entry missing a required field is a miss."""
    entry = _valid_entry()
    del entry["identity_count"]
    assert not _pose_cache_entry_matches(entry, "1:2", "v.h5")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("hdf5_frame_count", "10"),
        ("identity_count", 2.0),
        ("lixit_keypoints", True),
        ("has_cm_per_pixel", 1),
        ("static_objects", {"lixit": 1}),
        ("static_objects", "lixit"),
        ("static_objects", [1, 2]),
    ],
    ids=[
        "count-as-str",
        "count-as-float",
        "int-field-as-bool",
        "bool-field-as-int",
        "static-objects-as-dict",
        "static-objects-as-str",
        "static-objects-non-str-items",
    ],
)
def test_no_match_on_wrong_type(field, bad_value):
    """A parseable-but-wrong-typed field is a miss so the video is rescanned."""
    entry = _valid_entry()
    entry[field] = bad_value
    assert not _pose_cache_entry_matches(entry, "1:2", "v.h5")


# --- Project-level caching behavior ----------------------------------------


def test_second_load_uses_cache_no_scan(tmp_path, monkeypatch):
    """A second load of an unchanged project rescans no pose files."""
    project_dir = _make_project(tmp_path / "proj", ["v1.avi"])
    _open(project_dir)  # first load populates the cache
    assert (project_dir / CACHE_RELPATH).exists()

    calls = _spy_scan(monkeypatch)
    _open(project_dir)
    assert calls == []


def test_changed_pose_file_is_rescanned(tmp_path, monkeypatch):
    """A pose file whose token changed is rescanned on the next load."""
    project_dir = _make_project(tmp_path / "proj", ["v1.avi"])
    _open(project_dir)
    _bump_mtime(project_dir / "v1_pose_est_v3.h5")

    calls = _spy_scan(monkeypatch)
    _open(project_dir)
    assert calls == ["v1.avi"]


def test_new_video_only_scans_new(tmp_path, monkeypatch):
    """Adding a video rescans only the new video; cached ones are reused."""
    project_dir = _make_project(tmp_path / "proj", ["v1.avi"])
    _open(project_dir)
    _make_project(project_dir, ["v2.avi"])  # add a second video + pose

    calls = _spy_scan(monkeypatch)
    _open(project_dir)
    assert calls == ["v2.avi"]


def test_malformed_cache_entry_is_rescanned(tmp_path, monkeypatch):
    """A parseable cache whose entry has a wrong-typed field is rescanned."""
    project_dir = _make_project(tmp_path / "proj", ["v1.avi"])
    _open(project_dir)  # populate the cache

    cache_file = project_dir / CACHE_RELPATH
    data = json.loads(cache_file.read_text())
    # Corrupt a field's type: identity_count stored as a string.
    data["videos"]["v1.avi"]["identity_count"] = "not-an-int"
    cache_file.write_text(json.dumps(data))

    calls = _spy_scan(monkeypatch)
    _open(project_dir)
    assert calls == ["v1.avi"]


def test_use_cache_false_writes_no_cache(tmp_path):
    """With use_cache=False the project loads but persists no cache file."""
    project_dir = _make_project(tmp_path / "proj", ["v1.avi"])
    project = Project(
        project_dir,
        use_cache=False,
        enable_video_check=False,
        enable_session_tracker=False,
    )
    assert project.video_manager.videos == ["v1.avi"]
    assert not (project_dir / CACHE_RELPATH).exists()


def test_cached_load_matches_fresh_scan(tmp_path):
    """Values reconstructed from the cache match a full scan of the same project."""
    project_dir = _make_project(tmp_path / "proj", ["v1.avi", "v2.avi"])
    fresh = _open(project_dir)  # full scan (no cache yet)
    cached = _open(project_dir)  # cache hits

    assert cached.total_project_identities == fresh.total_project_identities
    assert cached.feature_manager.min_pose_version == fresh.feature_manager.min_pose_version
    assert cached.feature_manager.static_objects == fresh.feature_manager.static_objects
    assert cached.feature_manager.is_cm_unit == fresh.feature_manager.is_cm_unit
