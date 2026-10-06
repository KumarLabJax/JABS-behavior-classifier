"""The third review round: interop, fidelity and report-never-raise.

Each test here reproduced a reported failure before the fix.
"""

import json
import sys

import h5py
import numpy as np
import pytest

import jabs.io
from jabs.io.internal.pose_file import (
    Component,
    HistoryEntry,
    PoseFile,
    PoseFileError,
    Provenance,
    ProvenanceRecord,
    VideoInfo,
    read_component,
    read_pose_file,
    validate,
    write_pose_file,
)
from jabs.io.internal.pose_file.reader import attr_text
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


# --- interop with other HDF5 writers --------------------------------------


def test_a_numeric_format_attribute_is_rejected_not_recursed(tmp_path, sample_pose_file):
    """attr_text used to recurse forever on a non-string scalar."""
    assert attr_text(1) is None
    assert attr_text(np.int32(7)) is None
    path = tmp_path / "numeric.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r+") as h5:
        del h5.attrs["jabs_format"]
        h5.attrs["jabs_format"] = 1
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        assert "root_attrs" in _checks(path)
        with pytest.raises(PoseFileError):
            read_pose_file(path)
    finally:
        sys.setrecursionlimit(limit)


# --- string payloads keep their shape, dtype and representation ----------


def test_a_two_dimensional_string_component_round_trips(tmp_path, sample_pose_file):
    """`[str(v) for v in arr.tolist()]` stringified whole rows into one element."""
    names = np.array([["a", "b"], ["c", "d"]], dtype=object)
    component = Component(
        id="org.example.lab.labels",
        axes=("identity", "embedding"),
        data=names,
        missing={"policy": "none"},
    )
    path = tmp_path / "labels.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 1, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=1, width=4, height=4, fps=30.0),
            components=(component,),
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []
    back = read_pose_file(path).component("org.example.lab.labels")
    assert back.data.shape == (2, 2)
    assert back.data[1][0] == "c"


def test_byte_strings_are_decoded_not_repred(tmp_path):
    """b"a" became the literal text "b'a'"."""
    component = Component(
        id="org.example.lab.labels",
        axes=("identity",),
        data=np.array([b"a", b"b"], dtype=object),
        missing={"policy": "none"},
    )
    path = tmp_path / "bytes.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 1, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=1, width=4, height=4, fps=30.0),
            components=(component,),
        ),
        path,
    )
    assert list(read_component(path, "org.example.lab.labels")) == ["a", "b"]


def test_a_string_window_matches_a_whole_read(tmp_path):
    """The representation must not depend on whether frames was supplied."""
    component = Component(
        id="org.example.lab.notes",
        axes=("frame",),
        data=np.array(["w", "x", "y", "z"], dtype=object),
        missing={"policy": "none"},
    )
    path = tmp_path / "notes.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 1, "identity": 1},
            video=VideoInfo(frame_count=4, width=4, height=4, fps=30.0),
            components=(component,),
        ),
        path,
    )
    whole = read_component(path, "org.example.lab.notes")
    window = read_component(path, "org.example.lab.notes", frames=slice(1, 3))
    assert list(window) == list(whole[1:3]) == ["x", "y"]


# --- the public API forwards its options ---------------------------------


def test_public_save_forwards_created(tmp_path, sample_pose_file):
    """jabs.io.save discarded kwargs, so `created` silently used now."""
    path = tmp_path / "stamped.h5"
    jabs.io.save(sample_pose_file, path, created="2026-09-08T00:00:00Z")
    with h5py.File(path, "r") as h5:
        assert json.loads(h5["manifest"][()])["created"] == "2026-09-08T00:00:00Z"


def test_public_save_rejects_an_unsupported_option(tmp_path, sample_pose_file):
    """An unsupported option must fail visibly rather than be dropped."""
    with pytest.raises(TypeError):
        jabs.io.save(sample_pose_file, tmp_path / "x.h5", nonsense=True)


# --- provenance extras survive a round trip ------------------------------


def test_provenance_extras_round_trip(tmp_path, sample_pose_file):
    """Records, history entries and the document each take namespaced extras."""
    provenance = Provenance(
        records={
            "jabs.pose": ProvenanceRecord(
                producer="test",
                version="1",
                created="2026-09-08T00:00:00Z",
                extra={"org.example.lab": {"rig": "B2"}},
            )
        },
        history=(
            HistoryEntry(
                operation="infer",
                tool="test",
                version="1",
                time="2026-09-08T00:00:00Z",
                extra={"org.example.lab": {"operator": "kb"}},
            ),
        ),
        extra={"org.example.lab": {"protocol": "open-field"}},
    )
    points = sample_pose_file.component("jabs.pose.points")
    path = tmp_path / "prov.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=(points,),
            provenance=provenance,
        ),
        path,
    )
    back = read_pose_file(path).provenance
    assert back.extra == {"org.example.lab": {"protocol": "open-field"}}
    assert back.records["jabs.pose"].extra == {"org.example.lab": {"rig": "B2"}}
    assert back.history[0].extra == {"org.example.lab": {"operator": "kb"}}


def test_read_refuses_schema_invalid_provenance(written):
    """A record missing required fields leaked a KeyError from parsing."""
    with h5py.File(written, "r+") as h5:
        del h5["provenance"]
        h5.create_dataset(
            "provenance",
            data=json.dumps({"records": {"x": {}}, "history": []}),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    with pytest.raises(PoseFileError, match="provenance"):
        read_pose_file(written)


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


# --- the writer never publishes what its validator rejects ---------------


def test_the_writer_validates_the_finished_file(tmp_path, sample_pose_file):
    """File-level invariants can only be checked against the written file."""
    index_id = "jabs.dynamic_objects.fecal_boli.frame_index"
    index = Component(
        id=index_id,
        axes=("sample",),
        data=np.array([0, 1, 2], dtype=np.uint32),
        missing={"policy": "none"},
        units="frame",
        sparse_index=index_id,
    )
    path = tmp_path / "ok.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=(*sample_pose_file.components, index),
            provenance=sample_pose_file.provenance,
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []


def test_a_component_with_no_provenance_is_only_a_warning(tmp_path, sample_pose_file):
    """Requiring it would invalidate a third party's file (design goal 16)."""
    anonymous = Component(
        id="org.example.lab.thing",
        axes=("frame",),
        data=np.zeros(4, dtype=np.float32),
        missing={"policy": "nan"},
        units="unitless",
    )
    path = tmp_path / "anon.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 1, "identity": 1},
            video=VideoInfo(frame_count=4, width=4, height=4, fps=30.0),
            components=(anonymous,),
        ),
        path,
    )
    findings = validate(path)
    assert [f for f in findings if f.severity == "error"] == []
    assert any(f.check == "provenance_declared" for f in findings)
