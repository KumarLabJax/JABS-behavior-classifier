"""The validator must report, never raise, and must not miss what it claims.

Two classes of defect here. First, ``validate()`` is the tool aimed at
untrusted files, so a malformed one must produce a Finding rather than an
exception. Second, several checks were implemented weaker than the
specification text they cite, and passed files they should have failed.
"""

import json

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.validate import validate
from jabs.io.internal.pose_file.writer import write_pose_file


def _rewrite(path, dataset, payload):
    """Replace a JSON dataset in a written file."""
    with h5py.File(path, "r+") as h5:
        del h5[dataset]
        h5.create_dataset(dataset, data=payload, dtype=h5py.string_dtype(encoding="utf-8"))


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


def _errors(path):
    return [f for f in validate(path) if f.severity == "error"]


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


# --- validate() reports, it does not raise ---------------------------------


def test_missing_provenance_is_a_finding(written):
    """The document validate() already flagged must not then be dereferenced."""
    with h5py.File(written, "r+") as h5:
        del h5["provenance"]
    assert "provenance_schema" in _checks(written)


def test_malformed_provenance_json_is_a_finding(written):
    """Not-JSON must not escape as a JSONDecodeError."""
    _rewrite(written, "provenance", "{not json")
    assert "provenance_schema" in _checks(written)


def test_provenance_of_the_wrong_json_type_is_a_finding(written):
    """A JSON array where an object belongs used to raise AttributeError."""
    _rewrite(written, "provenance", "[]")
    assert "provenance_schema" in _checks(written)


def test_axes_arity_mismatch_does_not_crash(written):
    """The check exists to diagnose this file, so it must survive it."""
    _rewrite_manifest(written, lambda m: m["components"][0].update(axes=["frame"]))
    assert "axes_arity" in _checks(written) or "dtype_shape_match" in _checks(written)


def test_unreadable_schema_revision_is_a_finding(written):
    """A shape-(1,) version attribute is what legacy JABS writes."""
    with h5py.File(written, "r+") as h5:
        del h5.attrs["schema_revision"]
        h5.attrs["schema_revision"] = np.array([1], dtype=np.int32)
    findings = validate(written)
    assert all(f.check != "root_attrs" or f.severity == "warning" for f in findings)


def test_missing_schema_revision_still_checks_the_rest(written):
    """One missing attribute must not skip every payload and reference check."""
    with h5py.File(written, "r+") as h5:
        del h5.attrs["schema_revision"]
        del h5["/jabs/pose/confidence"]
    checks = _checks(written)
    assert "root_attrs" in checks
    assert "component_path_exists" in checks, "the pass stopped at the first problem"


def test_byte_string_format_attribute_is_accepted(tmp_path, sample_pose_file):
    """Fixed-length ASCII attributes are what other HDF5 APIs write."""
    path = tmp_path / "ascii_pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r+") as h5:
        del h5.attrs["jabs_format"]
        h5.attrs.create("jabs_format", np.bytes_(b"jabs.pose-file"))
    assert _errors(path) == []


def test_attachments_as_a_dataset_does_not_crash(written):
    """`'attachments' in h5` is a name test, not a group test."""
    with h5py.File(written, "r+") as h5:
        h5.create_dataset("attachments", data=np.zeros(3, dtype=np.uint8))
    validate(written)


# --- checks that were weaker than their specification ----------------------


def test_decreasing_unsigned_sparse_index_is_an_error(tmp_path, sample_pose_file):
    """np.diff wraps on unsigned, and the ADR prescribes uint32 for the index."""
    from jabs.io.internal.pose_file.types import Component, PoseFile

    index_id = "jabs.dynamic_objects.fecal_boli.frame_index"
    index = Component(
        id=index_id,
        axes=("sample",),
        data=np.array([2, 1, 0], dtype=np.uint32),
        missing={"policy": "none"},
        units="frame",
        sparse_index=index_id,
    )
    counts = Component(
        id="jabs.dynamic_objects.fecal_boli.counts",
        axes=("sample",),
        data=np.array([1, 1, 1], dtype=np.uint32),
        missing={"policy": "none"},
        units="unitless",
        sparse_index=index_id,
    )
    path = tmp_path / "decreasing.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=(*sample_pose_file.components, index, counts),
            provenance=sample_pose_file.provenance,
        ),
        path,
    )
    assert "sparse_index_valid" in _checks(path)


def test_a_shared_sparse_index_reports_once(tmp_path, sample_pose_file):
    """Two components referencing one bad index is one problem, not two."""
    from jabs.io.internal.pose_file.types import Component, PoseFile

    index_id = "jabs.dynamic_objects.fecal_boli.frame_index"
    index = Component(
        id=index_id,
        axes=("sample",),
        data=np.array([2, 1, 0], dtype=np.uint32),
        missing={"policy": "none"},
        units="frame",
        sparse_index=index_id,
    )
    others = tuple(
        Component(
            id=f"jabs.dynamic_objects.fecal_boli.{name}",
            axes=("sample",),
            data=np.array([1, 1, 1], dtype=np.uint32),
            missing={"policy": "none"},
            units="unitless",
            sparse_index=index_id,
        )
        for name in ("counts", "areas")
    )
    path = tmp_path / "shared.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=(*sample_pose_file.components, index, *others),
            provenance=sample_pose_file.provenance,
        ),
        path,
    )
    monotonic = [f for f in validate(path) if f.check == "sparse_index_valid"]
    assert len(monotonic) == 1, [f.message for f in monotonic]


def test_skeleton_edge_out_of_range_is_an_error(written):
    """The ADR's row has two clauses; only the reference one was implemented."""
    _rewrite_manifest(written, lambda m: m["skeletons"]["jabs.mouse12"].update(edges=[[0, 500]]))
    assert "skeleton_edge_range" in _checks(written)


def test_incompatible_mask_shape_is_an_error(written):
    """The ADR requires the reference to resolve to a compatible shape.

    A mask must align with the leading axes of what it masks: (4,2) masking
    (4,2,12,2) is fine -- one flag per slot per frame -- while a mask of higher
    rank than its target cannot align at all.
    """

    def mask_by_points(manifest):
        entry = next(c for c in manifest["components"] if c["id"] == "jabs.pose.slot_occupied")
        entry["missing"] = {"policy": "mask", "mask": "jabs.pose.points"}

    _rewrite_manifest(written, mask_by_points)
    assert "mask_reference" in _checks(written)


def test_a_leading_axis_mask_is_accepted(written):
    """The compatible case must not be reported."""

    def slot_mask(manifest):
        entry = next(c for c in manifest["components"] if c["id"] == "jabs.pose.points")
        entry["missing"] = {"policy": "mask", "mask": "jabs.pose.slot_occupied"}

    _rewrite_manifest(written, slot_mask)
    assert "mask_reference" not in _checks(written)


def test_sparse_index_must_have_a_sample_axis(written):
    """A frame-axis component cannot serve as a sample index."""

    def repoint(manifest):
        entry = next(c for c in manifest["components"] if c["id"] == "jabs.pose.slot_occupied")
        entry["axes"] = ["sample", "slot"]
        entry["sparse"] = {"index": "jabs.pose.confidence"}

    _rewrite_manifest(written, repoint)
    assert "sparse_index_valid" in _checks(written)


def test_frame_axis_length_must_match_dimensions(written):
    """The ADR: an axis named frame always has length dimensions.frame."""
    _rewrite_manifest(written, lambda m: m["dimensions"].update(frame=100000))
    assert "frame_axis_length" in _checks(written)


def test_dimensions_must_agree_with_video_frame_count(written):
    """Otherwise dimensions is decorative."""

    def diverge(manifest):
        manifest["dimensions"]["frame"] = 4
        manifest["video"]["frame_count"] = 999

    _rewrite_manifest(written, diverge)
    assert "dimensions_match_video" in _checks(written)


def test_manifest_format_must_agree_with_the_root_attributes(written):
    """Otherwise the forward-compatibility warning is bypassable."""
    _rewrite_manifest(written, lambda m: m.update(schema_revision=99))
    assert "manifest_matches_root" in _checks(written)
