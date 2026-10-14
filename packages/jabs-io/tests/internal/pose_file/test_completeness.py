"""Gaps between what the specification defines and what the code supported."""

import itertools
import json

import h5py
import numpy as np
import pytest

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.io.internal.pose_file import (
    Component,
    PoseFile,
    read_component,
    read_pose_file,
    validate,
    write_pose_file,
)
from jabs.io.internal.pose_file.skeletons import JABS_MOUSE12, jabs_mouse12


def _rewrite_manifest(path, mutate):
    """Replace a written file's manifest, leaving its arrays untouched."""
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5["manifest"][()])
        mutate(manifest)
        del h5["manifest"]
        h5.create_dataset(
            "manifest", data=json.dumps(manifest), dtype=h5py.string_dtype(encoding="utf-8")
        )


# --- string components -----------------------------------------------------


def test_external_ids_round_trip(tmp_path, sample_pose_file):
    """jabs.identity.external_ids is in the ADR's catalog and was unwritable."""
    external = Component(
        id="jabs.identity.external_ids",
        axes=("identity",),
        data=np.array(["mouse-a", "mouse-b"], dtype=object),
        missing={"policy": "none"},
        description="Display names for each identity.",
    )
    path = tmp_path / "named.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=(*sample_pose_file.components, external),
            provenance=sample_pose_file.provenance,
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []
    back = read_pose_file(path)
    assert list(back.component("jabs.identity.external_ids").data) == ["mouse-a", "mouse-b"]
    assert list(read_component(path, "jabs.identity.external_ids")) == ["mouse-a", "mouse-b"]


def test_string_component_declares_the_schema_dtype(tmp_path, sample_pose_file):
    """The manifest says "string", which is the name the schema admits."""
    external = Component(
        id="jabs.identity.external_ids",
        axes=("identity",),
        data=np.array(["a", "b"], dtype="<U8"),
        missing={"policy": "none"},
    )
    assert external.dtype == "string"
    path = tmp_path / "named.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            components=(external,),
        ),
        path,
    )
    with h5py.File(path, "r") as h5:
        entry = json.loads(h5["manifest"][()])["components"][0]
    assert entry["dtype"] == "string"


# --- the validated mappings cannot be mutated afterwards -------------------


def test_dimensions_cannot_be_mutated_past_validation(sample_pose_file):
    """identity <= slot is checked once; a live dict would let it be undone."""
    with pytest.raises(TypeError):
        sample_pose_file.dimensions["identity"] = 99


def test_missing_policy_cannot_be_mutated_past_validation(sample_pose_file):
    """The missing policy drives read dispatch, so it must stay put."""
    with pytest.raises(TypeError):
        sample_pose_file.component("jabs.pose.points").missing["policy"] = "none"


def test_axes_are_normalized_to_a_tuple():
    """A list would compare unequal to the tuple everything else uses."""
    component = Component(
        id="jabs.pose.slot_occupied",
        axes=["frame", "slot"],
        data=np.zeros((2, 1), dtype=bool),
        missing={"policy": "none"},
    )
    assert component.axes == ("frame", "slot")


# --- timestamps ------------------------------------------------------------


def test_unparseable_timestamp_is_an_error(tmp_path, sample_pose_file):
    """format: date-time is advisory in jsonschema without an extra package."""
    path = tmp_path / "stamped.h5"
    write_pose_file(sample_pose_file, path)
    _rewrite_manifest(path, lambda m: m.update(created="last Tuesday"))
    assert any(f.check == "timestamp_parseable" for f in validate(path))


def test_a_trailing_z_timestamp_is_accepted(tmp_path, sample_pose_file):
    """RFC 3339 Zulu form, which fromisoformat only accepts from 3.11."""
    path = tmp_path / "zulu.h5"
    write_pose_file(sample_pose_file, path, created="2026-09-04T12:00:00Z")
    assert [f for f in validate(path) if f.severity == "error"] == []


# --- reproducible output ---------------------------------------------------


def test_created_can_be_pinned(tmp_path, sample_pose_file):
    """Fixtures must be byte-reproducible, so `created` cannot be wall-clock."""
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    write_pose_file(sample_pose_file, first, created="2026-09-04T00:00:00Z")
    write_pose_file(sample_pose_file, second, created="2026-09-04T00:00:00Z")
    with h5py.File(first, "r") as a, h5py.File(second, "r") as b:
        assert json.loads(a["manifest"][()]) == json.loads(b["manifest"][()])


# --- the skeleton is derived, not transcribed ------------------------------


def test_the_jabs_skeleton_comes_from_jabs_core():
    """A hand-written copy had already drifted in case."""
    skeleton = jabs_mouse12()
    assert skeleton.body_parts == tuple(k.name for k in PoseEstimation.KeypointIndex)
    assert len(skeleton.body_parts) == 12
    assert JABS_MOUSE12 == "jabs.mouse12"


def test_the_derived_edges_cover_every_connected_segment():
    """Every adjacent pair in every polyline becomes an edge, once."""
    skeleton = jabs_mouse12()
    expected = set()
    for segment in PoseEstimation.FULL_CONNECTED_SEGMENTS:
        indexes = [int(point) for point in segment]
        for start, end in itertools.pairwise(indexes):
            expected.add(frozenset((start, end)))
    assert {frozenset(edge) for edge in skeleton.edges} == expected
    assert len(skeleton.edges) == len(expected), "an edge was emitted twice"


def test_the_derived_skeleton_validates_in_a_file(tmp_path, sample_pose_file):
    """And it is usable as the skeleton a real file declares."""
    points = sample_pose_file.component("jabs.pose.points")
    path = tmp_path / "derived.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons={JABS_MOUSE12: jabs_mouse12()},
            components=(
                Component(
                    id=points.id,
                    axes=points.axes,
                    data=points.data,
                    missing=points.missing,
                    units=points.units,
                    coord_order=points.coord_order,
                    skeleton=JABS_MOUSE12,
                ),
            ),
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []
