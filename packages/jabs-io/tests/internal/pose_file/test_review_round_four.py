"""Round four: constraints the prose implied and nothing enforced.

Alignment is not interpretability — a mask of the right shape and the wrong
dtype still cannot be applied — and a coordinate pair that is not a pair
cannot be read with `xy`.
"""

import json

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file import (
    Attachment,
    Component,
    PoseFile,
    VideoInfo,
    validate,
    write_pose_file,
)


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


def _entry(manifest, component_id):
    return next(c for c in manifest["components"] if c["id"] == component_id)


# --- a coordinate pair is a pair -----------------------------------------


def test_a_three_wide_coord_axis_is_refused_at_construction():
    """coord_order describes two values and nothing else."""
    with pytest.raises(ValueError, match="coord axis is 3 wide"):
        Component(
            id="org.example.lab.triples",
            axes=("frame", "coord"),
            data=np.zeros((4, 3), dtype=np.float32),
            missing={"policy": "nan"},
            units="pixel",
            coord_order="xy",
        )


def test_a_three_wide_coord_axis_is_an_error_in_a_file(tmp_path, sample_pose_file):
    """And reported for a file some other producer wrote."""
    path = tmp_path / "coord.h5"
    write_pose_file(sample_pose_file, path)
    _rewrite_manifest(
        path,
        lambda m: _entry(m, "jabs.pose.points").update(shape=[4, 2, 12, 3]),
    )
    assert "coord_axis_length" in _checks(path)


# --- a mask is boolean, a length is a bounded integer -------------------


def test_a_float_mask_is_refused_at_construction(sample_pose_file):
    """A float mask cannot say present or absent, whatever its shape."""
    confidence = sample_pose_file.component("jabs.pose.confidence")
    points = sample_pose_file.component("jabs.pose.points")
    with pytest.raises(ValueError, match="must be boolean"):
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=(
                Component(
                    id=points.id,
                    axes=points.axes,
                    data=points.data,
                    missing={"policy": "mask", "mask": confidence.id},
                    units=points.units,
                    coord_order=points.coord_order,
                    skeleton=points.skeleton,
                ),
                confidence,
            ),
        )


def test_a_float_mask_is_an_error_in_a_file(tmp_path, sample_pose_file):
    """Shape compatibility alone let this through."""
    path = tmp_path / "mask.h5"
    write_pose_file(sample_pose_file, path)
    _rewrite_manifest(
        path,
        lambda m: _entry(m, "jabs.pose.points").update(
            missing={"policy": "mask", "mask": "jabs.pose.confidence"}
        ),
    )
    assert "mask_reference" in _checks(path)


def test_a_negative_length_is_refused_at_construction(sample_pose_file):
    """A count cannot be negative."""
    counts = Component(
        id="org.example.lab.counts",
        axes=("frame",),
        data=np.array([-1, 0, 1, 2], dtype=np.int32),
        missing={"policy": "none"},
        units="unitless",
    )
    payload = Component(
        id="org.example.lab.values",
        axes=("frame", "slot"),
        data=np.zeros((4, 2), dtype=np.float32),
        missing={"policy": "length", "length": counts.id},
        units="unitless",
    )
    with pytest.raises(ValueError, match="outside"):
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
            components=(payload, counts),
        )


def test_an_out_of_range_length_is_refused(sample_pose_file):
    """A count larger than the axis it bounds cannot be applied."""
    counts = Component(
        id="org.example.lab.counts",
        axes=("frame",),
        data=np.array([0, 1, 2, 99], dtype=np.uint32),
        missing={"policy": "none"},
        units="unitless",
    )
    payload = Component(
        id="org.example.lab.values",
        axes=("frame", "slot"),
        data=np.zeros((4, 2), dtype=np.float32),
        missing={"policy": "length", "length": counts.id},
        units="unitless",
    )
    with pytest.raises(ValueError, match="outside"):
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
            components=(payload, counts),
        )


def test_a_valid_length_reference_is_accepted():
    """The dynamic-object pattern the ADR now specifies."""
    counts = Component(
        id="org.example.lab.counts",
        axes=("frame",),
        data=np.array([0, 1, 2, 2], dtype=np.uint32),
        missing={"policy": "none"},
        units="unitless",
    )
    payload = Component(
        id="org.example.lab.values",
        axes=("frame", "slot"),
        data=np.zeros((4, 2), dtype=np.float32),
        missing={"policy": "length", "length": counts.id},
        units="unitless",
    )
    pose_file = PoseFile(
        dimensions={"frame": 4, "slot": 2, "identity": 2},
        video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
        components=(payload, counts),
    )
    assert pose_file.component("org.example.lab.values").missing["policy"] == "length"


# --- offsets are unsigned integers ---------------------------------------


def _ragged_file(tmp_path, offsets_dtype):
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
    path = _ragged_file(tmp_path, np.uint64)
    assert [f for f in validate(path) if f.severity == "error"] == []


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_non_unsigned_offsets_are_an_error(tmp_path, dtype):
    """A float cannot index an array, and a signed offset admits negatives."""
    path = _ragged_file(tmp_path, dtype)
    assert "ragged_offsets" in _checks(path)


# --- attachments are checked in both directions -------------------------


def test_a_declared_attachment_with_no_payload_is_an_error(tmp_path, sample_pose_file):
    """A copy tool is required to preserve something that is not there."""
    path = tmp_path / "attach.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=sample_pose_file.components,
            provenance=sample_pose_file.provenance,
            attachments=(Attachment(path="/attachments/notes", data=np.zeros(3, dtype=np.uint8)),),
        ),
        path,
    )
    with h5py.File(path, "r+") as h5:
        del h5["/attachments/notes"]
    assert "attachment_path_exists" in _checks(path)


def test_an_attachment_outside_the_namespace_is_an_error(tmp_path, sample_pose_file):
    """`/attachments/` is where opaque payloads live."""
    path = tmp_path / "outside.h5"
    write_pose_file(sample_pose_file, path)
    _rewrite_manifest(path, lambda m: m.update(attachments=[{"path": "/elsewhere/notes"}]))
    with h5py.File(path, "r+") as h5:
        h5.create_dataset("/elsewhere/notes", data=np.zeros(3, dtype=np.uint8))
    assert "attachment_namespace" in _checks(path)


def test_a_twice_declared_attachment_is_an_error(tmp_path, sample_pose_file):
    """One payload, one declaration."""
    path = tmp_path / "twice.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=sample_pose_file.components,
            provenance=sample_pose_file.provenance,
            attachments=(Attachment(path="/attachments/notes", data=np.zeros(3, dtype=np.uint8)),),
        ),
        path,
    )
    _rewrite_manifest(
        path,
        lambda m: m.update(
            attachments=[{"path": "/attachments/notes"}, {"path": "/attachments/notes"}]
        ),
    )
    assert "attachment_unique" in _checks(path)
