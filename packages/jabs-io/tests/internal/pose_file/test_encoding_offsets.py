"""Non-dense offsets are validated even though this build cannot decode them.

Present, one-dimensional, unsigned, non-decreasing, and ending where they must.
"""

import json

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file import (
    validate,
)


def _checks(path):
    return {f.check for f in validate(path)}


# --- non-dense offsets are validated even though they cannot be decoded ---


def _ragged_file(tmp_path, group_offsets, instance_offsets, points=6):
    """Write a file by hand with a ragged component and given offsets."""
    path = tmp_path / "ragged.h5"
    manifest = {
        "format": "jabs.pose-file",
        "schema_revision": 1,
        "created": "2026-09-04T00:00:00Z",
        "dimensions": {"frame": 2, "slot": 1, "identity": 1},
        "video": {"frame_count": 2, "width": 8, "height": 8, "fps": 30.0},
        "components": [
            {
                "id": "jabs.segmentation.contours",
                "path": "/jabs/segmentation/contours",
                "axes": ["point", "coord"],
                "dtype": "int32",
                "shape": [points, 2],
                "units": "pixel",
                "coord_order": "xy",
                "encoding": {
                    "kind": "ragged",
                    "group_offsets": "/jabs/segmentation/group_offsets",
                    "instance_offsets": "/jabs/segmentation/instance_offsets",
                },
                "missing": {"policy": "none"},
            }
        ],
    }
    with h5py.File(path, "w") as h5:
        h5.attrs["jabs_format"] = "jabs.pose-file"
        h5.attrs["schema_revision"] = np.int32(1)
        text = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("manifest", data=json.dumps(manifest), dtype=text)
        h5.create_dataset(
            "provenance", data=json.dumps({"records": {}, "history": []}), dtype=text
        )
        h5.create_dataset(
            "/jabs/segmentation/contours", data=np.zeros((points, 2), dtype=np.int32)
        )
        if group_offsets is not None:
            h5.create_dataset(
                "/jabs/segmentation/group_offsets", data=np.array(group_offsets, dtype=np.uint64)
            )
        if instance_offsets is not None:
            h5.create_dataset(
                "/jabs/segmentation/instance_offsets",
                data=np.array(instance_offsets, dtype=np.uint64),
            )
    return path


def test_well_formed_ragged_offsets_validate(tmp_path):
    """A conforming ragged file is conforming even though we cannot decode it."""
    path = _ragged_file(tmp_path, group_offsets=[0, 3, 6], instance_offsets=[0, 1, 2])
    assert [f for f in validate(path) if f.severity == "error"] == []


def test_missing_ragged_offsets_are_an_error(tmp_path):
    """The ADR requires the offsets to exist."""
    path = _ragged_file(tmp_path, group_offsets=None, instance_offsets=[0, 1, 2])
    assert "ragged_offsets" in _checks(path)


def test_decreasing_ragged_offsets_are_an_error(tmp_path):
    """Non-decreasing, and uint64 must not wrap the comparison."""
    path = _ragged_file(tmp_path, group_offsets=[0, 6, 3], instance_offsets=[0, 1, 2])
    assert "ragged_offsets" in _checks(path)


def test_ragged_offsets_must_end_at_the_payload_length(tmp_path):
    """A terminal offset that is not the payload length loses data silently."""
    path = _ragged_file(tmp_path, group_offsets=[0, 3, 4], instance_offsets=[0, 1, 2])
    assert "ragged_offsets" in _checks(path)


def test_instance_offsets_must_have_frame_times_slot_plus_one_entries(tmp_path):
    """frame*slot+1 is what addresses every instance."""
    path = _ragged_file(tmp_path, group_offsets=[0, 3, 6], instance_offsets=[0, 2])
    assert "ragged_offsets" in _checks(path)


# --- offsets are unsigned integers ---------------------------------------


def _ragged_file_with_offsets_dtype(tmp_path, offsets_dtype):
    """A ragged component whose offsets use the given dtype."""
    path = tmp_path / f"ragged_{np.dtype(offsets_dtype).name}.h5"
    manifest = {
        "format": "jabs.pose-file",
        "schema_revision": 1,
        "created": "2026-09-08T00:00:00Z",
        "dimensions": {"frame": 2, "slot": 1, "identity": 1},
        "video": {"frame_count": 2, "width": 8, "height": 8, "fps": 30.0},
        "components": [
            {
                "id": "jabs.segmentation.contours",
                "path": "/jabs/segmentation/contours",
                "axes": ["point", "coord"],
                "dtype": "int32",
                "shape": [6, 2],
                "units": "pixel",
                "coord_order": "xy",
                "encoding": {
                    "kind": "ragged",
                    "group_offsets": "/jabs/segmentation/group_offsets",
                    "instance_offsets": "/jabs/segmentation/instance_offsets",
                },
                "missing": {"policy": "none"},
            }
        ],
    }
    with h5py.File(path, "w") as h5:
        h5.attrs["jabs_format"] = "jabs.pose-file"
        h5.attrs["schema_revision"] = np.int32(1)
        text = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("manifest", data=json.dumps(manifest), dtype=text)
        h5.create_dataset(
            "provenance", data=json.dumps({"records": {}, "history": []}), dtype=text
        )
        h5.create_dataset("/jabs/segmentation/contours", data=np.zeros((6, 2), dtype=np.int32))
        h5.create_dataset(
            "/jabs/segmentation/group_offsets", data=np.array([0, 3, 6], dtype=offsets_dtype)
        )
        h5.create_dataset(
            "/jabs/segmentation/instance_offsets", data=np.array([0, 1, 2], dtype=offsets_dtype)
        )
    return path


def test_unsigned_offsets_validate(tmp_path):
    """uint64 is what the encoding text requires."""
    path = _ragged_file_with_offsets_dtype(tmp_path, np.uint64)
    assert [f for f in validate(path) if f.severity == "error"] == []


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_non_unsigned_offsets_are_an_error(tmp_path, dtype):
    """A float cannot index an array, and a signed offset admits negatives."""
    path = _ragged_file_with_offsets_dtype(tmp_path, dtype)
    assert "ragged_offsets" in _checks(path)
