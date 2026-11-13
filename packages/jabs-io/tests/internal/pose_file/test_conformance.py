"""The fixture corpus that any independent implementation can test against.

Regenerate the fixtures with
``uv run python packages/jabs-io/tests/data/pose-format/generate.py``.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.reader import read_component, read_pose_file
from jabs.io.internal.pose_file.validate import validate

DATA = Path(__file__).parents[2] / "data" / "pose-format"

VALID = [
    "valid-minimal.h5",
    "valid-full.h5",
    "valid-sparse.h5",
    "valid-ragged.h5",
    "valid-ragged-alternate.h5",
    "valid-rle.h5",
]
INVALID = {
    "invalid-ragged-level.h5": "ragged_offsets",
    "invalid-shape-mismatch.h5": "dtype_shape_match",
    "invalid-missing-payload.h5": "component_path_exists",
    "invalid-dangling-mask.h5": "mask_reference",
    "invalid-keypoint-axis.h5": "keypoint_axis_length",
    "invalid-not-a-pose-file.h5": "root_attrs",
}


@pytest.mark.parametrize("name", VALID)
def test_valid_fixtures_have_no_errors(name):
    """Every valid fixture conforms."""
    errors = [f for f in validate(DATA / name) if f.severity == "error"]
    assert errors == [], f"{name} should be valid, got {[f.message for f in errors]}"


@pytest.mark.parametrize("name,expected_check", sorted(INVALID.items()))
def test_invalid_fixtures_report_their_check(name, expected_check):
    """Every invalid fixture fails the specific rule it was built to break."""
    findings = validate(DATA / name)
    assert any(f.check == expected_check and f.severity == "error" for f in findings), (
        f"{name} should report {expected_check}; got {sorted({f.check for f in findings})}"
    )


def test_fixture_corpus_is_complete():
    """The corpus on disk covers everything the suite names."""
    present = {p.name for p in DATA.glob("*.h5")}
    assert present >= set(VALID) | set(INVALID)


def test_full_fixture_carries_a_foreign_component():
    """Extensibility is demonstrated, not just asserted in prose."""
    pose_file = read_pose_file(DATA / "valid-full.h5")
    foreign = pose_file.component("org.example.lab.whisker_angle")
    assert foreign.axes == ("frame", "slot")
    assert foreign.path == "/org.example.lab/whisker_angle"
    assert foreign.units == "radian"


def test_a_foreign_component_can_be_windowed_without_understanding_it():
    """The point of declared axes: generic tooling can subset unknown data."""
    window = read_component(
        DATA / "valid-full.h5", "org.example.lab.whisker_angle", frames=slice(2, 5)
    )
    assert window.shape[0] == 3


def test_undeclared_attachment_is_only_a_warning():
    """An opaque payload nobody declared does not make the file invalid."""
    findings = validate(DATA / "valid-full.h5")
    assert [f for f in findings if f.severity == "error"] == []
    assert any(f.check == "attachment_undeclared" for f in findings)


def test_sparse_fixture_maps_samples_to_frames():
    """A sparse component's index says which frames it describes."""
    pose_file = read_pose_file(DATA / "valid-sparse.h5")
    index = pose_file.component("jabs.dynamic_objects.fecal_boli.frame_index")
    counts = pose_file.component("jabs.dynamic_objects.fecal_boli.counts")
    assert index.axes == ("sample",)
    assert index.units == "frame"
    np.testing.assert_array_equal(index.data, np.array([0, 3, 6], dtype=np.uint32))
    assert counts.sparse_index == index.id
    assert counts.data.shape == index.data.shape


def _decode_ragged(h5: h5py.File, contours: dict, flags: dict, frames: int, slots: int):
    """Decode a leaf/group ragged pair exactly as ADR 0002 spells it out.

    Leaf level: group g spans payload[group_offsets[g] : group_offsets[g + 1]],
    and instance (f, s) owns groups [instance_offsets[i], instance_offsets[i + 1]).
    Group level: instance (f, s) owns payload[instance_offsets[i] :
    instance_offsets[i + 1]] directly.
    """
    points = h5[contours["path"]][()]
    flag_values = h5[flags["path"]][()]
    group = h5[contours["encoding"]["group_offsets"]][()].astype(np.int64)
    instance = h5[contours["encoding"]["instance_offsets"]][()].astype(np.int64)
    decoded = {}
    for f in range(frames):
        for s in range(slots):
            i = f * slots + s
            first, last = instance[i], instance[i + 1]
            outlines = [points[group[g] : group[g + 1]].tolist() for g in range(first, last)]
            decoded[f, s] = (outlines, flag_values[first:last].tolist())
    return decoded


def _decode_dense(pose_file, frames: int, slots: int):
    data = pose_file.component("jabs.segmentation.contours").data
    count = pose_file.component("jabs.segmentation.contour_count").data
    length = pose_file.component("jabs.segmentation.contour_length").data
    flag = pose_file.component("jabs.segmentation.external_flag").data
    decoded = {}
    for f in range(frames):
        for s in range(slots):
            n = int(count[f, s])
            outlines = [data[f, s, c, : int(length[f, s, c])].tolist() for c in range(n)]
            decoded[f, s] = (outlines, flag[f, s, :n].tolist())
    return decoded


def test_the_ragged_alternate_decodes_to_its_dense_baseline():
    """The fixture an independent decoder checks itself against.

    It includes an instance whose every contour has one point, where the leaf
    and group payloads of that instance are the same length: a decoder that
    guesses the level instead of reading it gets that instance wrong.
    """
    path = DATA / "valid-ragged-alternate.h5"
    pose_file = read_pose_file(path)
    frames, slots = pose_file.dimensions["frame"], pose_file.dimensions["slot"]
    with h5py.File(path, "r") as h5:
        manifest = json.loads(h5["manifest"][()])
        by_id = {c["id"]: c for c in manifest["components"]}
        contours = by_id["jabs.segmentation.contours"]["alternates"][0]
        flags = by_id["jabs.segmentation.external_flag"]["alternates"][0]
        assert (contours["encoding"]["level"], flags["encoding"]["level"]) == ("leaf", "group")
        ragged = _decode_ragged(h5, contours, flags, frames, slots)
    assert ragged == _decode_dense(pose_file, frames, slots)
    one_point = [
        k for k, (outlines, _) in ragged.items() if outlines and all(len(o) == 1 for o in outlines)
    ]
    assert one_point, "the fixture must keep an instance of one-point contours"


def test_the_ragged_alternate_is_recorded_as_an_encode():
    """Adding an alternate writes a derived file, and its history says so."""
    pose_file = read_pose_file(DATA / "valid-ragged-alternate.h5")
    encode = [e for e in pose_file.provenance.history if e.operation == "encode"]
    assert len(encode) == 1
    assert "jabs.segmentation.contours:alternate:ragged" in encode[0].synthesized
