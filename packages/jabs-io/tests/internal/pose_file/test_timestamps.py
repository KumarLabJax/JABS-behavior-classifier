"""The schemas' date-time format is enforced.

As RFC 3339, which is stricter than ``fromisoformat``'s reading.
"""

import json

import h5py
import pytest

from jabs.io.internal.pose_file import (
    validate,
    write_pose_file,
)
from jabs.io.internal.pose_file.schema import validate_manifest


def _rewrite_manifest(path, mutate):
    """Replace a written file's manifest, leaving its arrays untouched."""
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5["manifest"][()])
        mutate(manifest)
        del h5["manifest"]
        h5.create_dataset(
            "manifest", data=json.dumps(manifest), dtype=h5py.string_dtype(encoding="utf-8")
        )


def _checks(path):
    return {f.check for f in validate(path)}


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


# --- the schema helpers enforce what the schemas declare -------------------


def test_validate_manifest_rejects_a_bad_timestamp(minimal_manifest):
    """format: date-time is annotation-only unless a checker is supplied."""
    minimal_manifest["created"] = "not-a-date"
    assert validate_manifest(minimal_manifest) != []


def test_validate_manifest_accepts_a_real_timestamp(minimal_manifest):
    """Including the RFC 3339 Zulu form."""
    minimal_manifest["created"] = "2026-09-04T12:00:00Z"
    assert validate_manifest(minimal_manifest) == []


def test_a_bad_timestamp_does_not_hide_structural_problems(tmp_path, sample_pose_file):
    """A format failure must not abort the pass the way a schema failure does."""
    path = tmp_path / "stamped.h5"
    write_pose_file(sample_pose_file, path)
    _rewrite_manifest(path, lambda m: m.update(created="last Tuesday"))
    with h5py.File(path, "r+") as h5:
        del h5["/jabs/pose/confidence"]
    checks = _checks(path)
    assert "timestamp_parseable" in checks
    assert "component_path_exists" in checks, "the timestamp hid the missing payload"


# --- RFC 3339 is stricter than fromisoformat -----------------------------


@pytest.mark.parametrize("value", ["2026-09-04", "2026-09-04T12:00:00"])
def test_a_date_without_time_or_offset_is_rejected(minimal_manifest, value):
    """A date is not a date-time, and a local time is not a point in time."""
    minimal_manifest["created"] = value
    assert validate_manifest(minimal_manifest) != []


@pytest.mark.parametrize("value", ["2026-09-04T12:00:00Z", "2026-09-04T12:00:00+00:00"])
def test_a_real_rfc3339_timestamp_is_accepted(minimal_manifest, value):
    """Both the Zulu and explicit-offset forms."""
    minimal_manifest["created"] = value
    assert validate_manifest(minimal_manifest) == []
