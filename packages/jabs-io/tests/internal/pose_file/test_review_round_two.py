"""The four findings from the second review that survived the first fix pass.

The other ten were already closed; these were not, and each is verified here
against the behaviour the specification asks for.
"""

import json

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file import (
    Component,
    PoseFile,
    VideoInfo,
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


# --- the writer cannot emit a file the validator rejects -------------------


def test_a_frame_axis_must_match_dimensions_at_construction(sample_pose_file):
    """Previously the writer emitted this and validate() then flagged it."""
    with pytest.raises(ValueError, match="frame axis is 5 but dimensions.frame is 4"):
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
            components=(
                Component(
                    id="jabs.pose.slot_occupied",
                    axes=("frame", "slot"),
                    data=np.zeros((5, 2), dtype=bool),
                    missing={"policy": "none"},
                ),
            ),
        )


def test_a_slot_axis_must_match_dimensions_at_construction():
    """The same reasoning covers every axis the file gives a size."""
    with pytest.raises(ValueError, match="slot axis is 9 but dimensions.slot is 2"):
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
            components=(
                Component(
                    id="jabs.pose.slot_occupied",
                    axes=("frame", "slot"),
                    data=np.zeros((4, 9), dtype=bool),
                    missing={"policy": "none"},
                ),
            ),
        )


def test_dimensions_must_agree_with_video_at_construction():
    """dimensions.frame and video.frame_count are the same fact."""
    with pytest.raises(ValueError, match="disagrees with video.frame_count"):
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=99, width=8, height=8, fps=30.0),
        )


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


# --- the declared layout is compared in full ------------------------------


def test_a_wrong_declared_chunk_shape_is_a_warning(tmp_path, sample_pose_file):
    """chunks and compression_opts are part of the layout schema."""
    seg = Component(
        id="jabs.segmentation.contours",
        axes=("frame", "slot"),
        data=np.zeros((4, 2), dtype=np.int32),
        missing={"policy": "none"},
    )
    path = tmp_path / "layout.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
            components=(seg,),
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []
    _rewrite_manifest(
        path,
        lambda m: m["components"][0]["layout"].update(chunks=[999, 999]),
    )
    assert "layout_matches_file" in _checks(path)


def test_a_wrong_declared_compression_level_is_a_warning(tmp_path):
    """A manifest claiming gzip 9 over a level-1 dataset is inaccurate."""
    seg = Component(
        id="jabs.segmentation.contours",
        axes=("frame", "slot"),
        data=np.zeros((4, 2), dtype=np.int32),
        missing={"policy": "none"},
    )
    path = tmp_path / "level.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
            components=(seg,),
        ),
        path,
    )
    _rewrite_manifest(path, lambda m: m["components"][0]["layout"].update(compression_opts=9))
    assert "layout_matches_file" in _checks(path)
