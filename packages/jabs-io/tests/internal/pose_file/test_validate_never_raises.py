"""validate() reports every problem as a Finding.

Including on a file it cannot open and on records the schema rejects.
"""

import json

import h5py
import pytest

from jabs.io.internal.pose_file import (
    validate,
    write_pose_file,
)


def _checks(path):
    return {f.check for f in validate(path)}


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


# --- validate() reports, even on a file it cannot open -------------------


def test_a_non_hdf5_file_is_a_finding(tmp_path):
    """Opening was outside the error handling, so OSError escaped."""
    path = tmp_path / "garbage.h5"
    path.write_bytes(b"not hdf5 at all")
    findings = validate(path)
    assert any(f.check == "file_readable" for f in findings)


def test_schema_invalid_records_do_not_crash(written):
    """`set(1)` raised TypeError on a JSON object with non-dict records."""
    with h5py.File(written, "r+") as h5:
        del h5["provenance"]
        h5.create_dataset(
            "provenance",
            data=json.dumps({"records": 1, "history": []}),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    assert "provenance_schema" in _checks(written)
