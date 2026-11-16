"""Attachments are checked in both directions.

A declared attachment must exist, live under /attachments/ and be declared once.
"""

import json

import h5py
import numpy as np

from jabs.io.internal.pose_file import (
    Attachment,
    PoseFile,
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
