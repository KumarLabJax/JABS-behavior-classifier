"""Invariants enforced at construction, and the matching checks on files.

The typed objects refuse what the validator would reject, so the writer cannot emit
it; the validator applies the same rules to files another producer wrote.
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


def _entry(manifest, component_id):
    return next(c for c in manifest["components"] if c["id"] == component_id)


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


# --- checks that were narrower than the invariant ------------------------


def test_every_named_axis_is_checked_against_dimensions(written):
    """Only the frame axis was compared, though PoseFile checks them all."""
    _rewrite_manifest(written, lambda m: m["dimensions"].update(slot=9))
    checks = _checks(written)
    assert "axis_length" in checks or "identity_le_slot" in checks


def test_a_same_length_but_wrong_axis_mask_is_an_error(tmp_path, sample_pose_file):
    """Matching lengths are not alignment."""
    frames = Component(
        id="org.example.lab.per_frame",
        axes=("frame",),
        data=np.zeros(4, dtype=np.float32),
        missing={"policy": "nan"},
        units="unitless",
    )
    identities = Component(
        id="org.example.lab.per_identity",
        axes=("identity",),
        data=np.zeros(4, dtype=bool),
        missing={"policy": "none"},
    )
    path = tmp_path / "misaligned.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 4, "identity": 4},
            video=VideoInfo(frame_count=4, width=4, height=4, fps=30.0),
            components=(frames, identities),
        ),
        path,
    )
    _rewrite_manifest(
        path,
        lambda m: next(
            c for c in m["components"] if c["id"] == "org.example.lab.per_frame"
        ).update(missing={"policy": "mask", "mask": "org.example.lab.per_identity"}),
    )
    assert "mask_reference" in _checks(path)


def test_a_float_sparse_index_is_an_error(written):
    """Casting first truncated [0.5, 1.5] into apparently valid frames."""

    def repoint(manifest):
        entry = next(c for c in manifest["components"] if c["id"] == "jabs.pose.slot_occupied")
        entry["axes"] = ["sample", "slot"]
        entry["sparse"] = {"index": "org.example.lab.float_index"}
        manifest["components"].append(
            {
                "id": "org.example.lab.float_index",
                "path": "/org.example.lab/float_index",
                "axes": ["sample"],
                "dtype": "float32",
                "shape": [4],
                "units": "frame",
                "encoding": {"kind": "dense"},
                "missing": {"policy": "none"},
                "sparse": {"index": "org.example.lab.float_index"},
            }
        )

    _rewrite_manifest(written, repoint)
    with h5py.File(written, "r+") as h5:
        h5.create_dataset(
            "/org.example.lab/float_index", data=np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        )
    assert "sparse_index_valid" in _checks(written)


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
