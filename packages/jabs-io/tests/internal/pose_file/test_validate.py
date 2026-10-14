"""Tests for the pose file validator."""

import json

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.validate import validate
from jabs.io.internal.pose_file.writer import write_pose_file


def _rewrite_manifest(path, mutate):
    """Replace the manifest in a written file, leaving the arrays untouched."""
    with h5py.File(path, "r+") as h5:
        manifest = json.loads(h5["manifest"][()])
        mutate(manifest)
        del h5["manifest"]
        h5.create_dataset(
            "manifest",
            data=json.dumps(manifest),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def _errors(path):
    return [f for f in validate(path) if f.severity == "error"]


def _checks(path):
    return {f.check for f in validate(path)}


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


def test_a_written_file_is_clean(written):
    """Whatever the writer produces must validate."""
    assert _errors(written) == []


def test_not_a_pose_file_is_an_error(tmp_path):
    """A legacy file fails at the first check rather than crashing."""
    legacy = tmp_path / "vid_pose_est_v6.h5"
    with h5py.File(legacy, "w") as h5:
        group = h5.create_group("poseest")
        group.attrs["version"] = np.array([6, 0], dtype=np.uint16)
    assert "root_attrs" in _checks(legacy)


def test_schema_violation_is_an_error(written):
    """A manifest the schema rejects is reported as such."""
    _rewrite_manifest(written, lambda m: m.pop("video"))
    assert "manifest_schema" in _checks(written)


def test_shape_disagreement_is_an_error(written):
    """The manifest is a second source of truth, so it can disagree."""
    _rewrite_manifest(written, lambda m: m["components"][0].update(shape=[99, 2, 12, 2]))
    assert "dtype_shape_match" in _checks(written)


def test_missing_payload_is_an_error(written):
    """A declared component must actually be in the file."""
    with h5py.File(written, "r+") as h5:
        del h5["/jabs/pose/confidence"]
    assert "component_path_exists" in _checks(written)


def test_dangling_mask_reference_is_an_error(written):
    """A missing policy that names a component nobody wrote is undecodable."""
    _rewrite_manifest(
        written,
        lambda m: m["components"][0].update(missing={"policy": "mask", "mask": "jabs.nope.mask"}),
    )
    assert "mask_reference" in _checks(written)


def test_dangling_provenance_reference_is_an_error(written):
    """So is a component pointing at a provenance record that does not exist."""
    _rewrite_manifest(written, lambda m: m["components"][0].update(provenance="not.a.record"))
    assert "provenance_reference" in _checks(written)


def test_keypoint_axis_must_match_the_skeleton(written):
    """A 12-wide keypoint axis cannot belong to a 2-keypoint skeleton."""

    def mutate(manifest):
        manifest["skeletons"]["jabs.mouse12"]["body_parts"] = ["NOSE", "TIP_TAIL"]
        manifest["skeletons"]["jabs.mouse12"]["edges"] = [[0, 1]]

    _rewrite_manifest(written, mutate)
    assert "keypoint_axis_length" in _checks(written)


def test_identity_may_not_exceed_slot(written):
    """Slots hold identities plus unassigned instances, never fewer."""
    _rewrite_manifest(written, lambda m: m["dimensions"].update(identity=9))
    assert "identity_le_slot" in _checks(written)


def test_null_video_dimensions_are_a_warning_not_an_error(
    tmp_path, sample_pose_file_no_dimensions
):
    """A converted file with no video is legal, and says so."""
    path = tmp_path / "c_pose.h5"
    write_pose_file(sample_pose_file_no_dimensions, path)
    assert _errors(path) == []
    assert "video_dimensions_null" in _checks(path)


def test_undeclared_attachment_is_a_warning(written):
    """An attachment nobody declared is worth mentioning, not refusing."""
    with h5py.File(written, "r+") as h5:
        h5.create_dataset("/attachments/notes", data=np.frombuffer(b"hello", dtype=np.uint8))
    assert _errors(written) == []
    assert "attachment_undeclared" in _checks(written)


def test_layout_disagreement_is_a_warning(written):
    """Declared storage that does not match the file is reported."""
    _rewrite_manifest(
        written,
        lambda m: m["components"][0].update(layout={"storage": "chunked", "compression": "gzip"}),
    )
    assert _errors(written) == []
    assert "layout_matches_file" in _checks(written)


def test_findings_are_readable(written):
    """A finding names its check and explains itself."""
    _rewrite_manifest(written, lambda m: m["components"][0].update(shape=[99, 2, 12, 2]))
    finding = next(f for f in validate(written) if f.check == "dtype_shape_match")
    assert finding.severity == "error"
    assert "jabs.pose.points" in finding.message
