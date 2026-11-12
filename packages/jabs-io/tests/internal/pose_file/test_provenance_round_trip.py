"""Provenance survives a round trip intact.

A schema-invalid document is refused on read rather than half-parsed.
"""

import json

import h5py
import pytest

from jabs.io.internal.pose_file import (
    HistoryEntry,
    PoseFile,
    PoseFileError,
    Provenance,
    ProvenanceRecord,
    read_pose_file,
    write_pose_file,
)


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


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
