"""The writer must never destroy a file it then fails to replace.

Every test here writes a sentinel to the destination first and asserts it
survives. ``h5py.File(path, "w")`` truncates at open, so any failure between
the open and a successful close costs the caller their existing file — and the
schema validations that run before the open do not catch everything.
"""

import json

import h5py
import numpy as np
import pytest

import jabs.io
from jabs.io.internal.pose_file import (
    Component,
    PoseFile,
    Provenance,
    ProvenanceRecord,
    read_pose_file,
    validate,
    write_pose_file,
)

SENTINEL = b"irreplaceable prior results"


def _with_provenance_parameters(sample_pose_file: PoseFile, parameters: dict) -> PoseFile:
    """The sample file with one provenance record's parameters replaced."""
    record = sample_pose_file.provenance.records["jabs.pose"]
    return PoseFile(
        dimensions=sample_pose_file.dimensions,
        video=sample_pose_file.video,
        skeletons=sample_pose_file.skeletons,
        components=sample_pose_file.components,
        provenance=Provenance(
            records={
                "jabs.pose": ProvenanceRecord(
                    producer=record.producer,
                    version=record.version,
                    created=record.created,
                    parameters=parameters,
                )
            },
            history=sample_pose_file.provenance.history,
        ),
    )


def test_unserializable_provenance_leaves_the_destination_intact(tmp_path, sample_pose_file):
    """np.int64 satisfies the schema and breaks json.dumps."""
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)
    # The schema types `parameters` as a bare object, and np.int64 is not an int
    # subclass, so validation passes and serialization fails.
    broken = _with_provenance_parameters(sample_pose_file, {"seed": np.int64(42)})
    with pytest.raises((TypeError, ValueError)):
        write_pose_file(broken, path)
    assert path.read_bytes() == SENTINEL


def test_schema_failure_leaves_the_destination_intact(tmp_path, sample_pose_file):
    """A skeleton id the schema rejects but PoseFile accepts."""
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)
    points = sample_pose_file.component("jabs.pose.points")
    relabeled = Component(
        id=points.id,
        axes=points.axes,
        data=points.data,
        missing=points.missing,
        units=points.units,
        coord_order=points.coord_order,
        skeleton="mouse",
        provenance=points.provenance,
    )
    # PoseFile only checks that the reference resolves; the schema additionally
    # requires skeleton ids to be namespaced, so "mouse" fails at write time.
    broken = PoseFile(
        dimensions=sample_pose_file.dimensions,
        video=sample_pose_file.video,
        skeletons={"mouse": sample_pose_file.skeletons["jabs.mouse12"]},
        components=(relabeled,),
        provenance=sample_pose_file.provenance,
    )
    with pytest.raises(ValueError, match="invalid pose file"):
        write_pose_file(broken, path)
    assert path.read_bytes() == SENTINEL


def test_component_path_collision_is_refused_before_writing(tmp_path, sample_pose_file):
    """One component's path may not be a parent of another's."""
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)
    points = sample_pose_file.component("jabs.pose.points")
    nested = Component(
        id="jabs.pose.points.confidence",
        axes=("frame", "slot"),
        data=np.zeros((4, 2), dtype=np.float32),
        missing={"policy": "none"},
        provenance="jabs.pose",
    )
    broken = PoseFile(
        dimensions=sample_pose_file.dimensions,
        video=sample_pose_file.video,
        skeletons=sample_pose_file.skeletons,
        components=(points, nested),
        provenance=sample_pose_file.provenance,
    )
    with pytest.raises(ValueError, match="collide|prefix|conflict"):
        write_pose_file(broken, path)
    assert path.read_bytes() == SENTINEL


def test_successful_write_replaces_the_destination(tmp_path, sample_pose_file):
    """The happy path still overwrites, and leaves no temporary behind."""
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)
    write_pose_file(sample_pose_file, path)
    assert path.read_bytes() != SENTINEL
    assert [f for f in validate(path) if f.severity == "error"] == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["prior.h5"]


def test_public_save_and_load_round_trip(tmp_path, sample_pose_file):
    """jabs.io.save is the package's public write API and must route correctly."""
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)
    jabs.io.save(sample_pose_file, path)
    assert [f for f in validate(path) if f.severity == "error"] == []
    back = jabs.io.load(path, PoseFile)
    np.testing.assert_array_equal(
        back.component("jabs.pose.points").data,
        sample_pose_file.component("jabs.pose.points").data,
    )


def test_public_save_failure_leaves_the_destination_intact(tmp_path, sample_pose_file):
    """And it must not truncate when it cannot write."""
    path = tmp_path / "prior.h5"
    path.write_bytes(SENTINEL)
    broken = _with_provenance_parameters(sample_pose_file, {"seed": np.int64(1)})
    with pytest.raises((TypeError, ValueError)):
        jabs.io.save(broken, path)
    assert path.read_bytes() == SENTINEL


def test_manifest_is_serializable_with_numpy_scalars(tmp_path, sample_pose_file):
    """Numpy scalars in dimensions and video must not break serialization."""
    numpy_dimensions = {k: np.int64(v) for k, v in sample_pose_file.dimensions.items()}
    pose_file = PoseFile(
        dimensions=numpy_dimensions,
        video=sample_pose_file.video,
        skeletons=sample_pose_file.skeletons,
        components=sample_pose_file.components,
        provenance=sample_pose_file.provenance,
    )
    path = tmp_path / "numpy_pose.h5"
    write_pose_file(pose_file, path)
    with h5py.File(path, "r") as h5:
        manifest = json.loads(h5["manifest"][()])
    assert manifest["dimensions"]["frame"] == 4
    assert read_pose_file(path).dimensions["frame"] == 4
