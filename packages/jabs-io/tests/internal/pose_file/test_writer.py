"""Tests for writing pose files."""

import json

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.schema import validate_manifest, validate_provenance
from jabs.io.internal.pose_file.types import Component, PoseFile
from jabs.io.internal.pose_file.writer import write_pose_file


def test_root_attributes(tmp_path, sample_pose_file):
    """A reader identifies the format from the root attributes alone."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r") as h5:
        assert h5.attrs["jabs_format"] == "jabs.pose-file"
        assert int(h5.attrs["schema_revision"]) == 1


def test_manifest_and_provenance_are_valid_json_documents(tmp_path, sample_pose_file):
    """Both documents on disk satisfy the shipped schemas."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r") as h5:
        manifest = json.loads(h5["manifest"][()])
        provenance = json.loads(h5["provenance"][()])
    assert validate_manifest(manifest) == []
    assert validate_provenance(provenance) == []


def test_payload_lives_at_the_declared_path(tmp_path, sample_pose_file):
    """One canonical place a validator looks for a component's data."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r") as h5:
        manifest = json.loads(h5["manifest"][()])
        for entry in manifest["components"]:
            dataset = h5[entry["path"]]
            assert list(dataset.shape) == entry["shape"]
            assert dataset.dtype.name == entry["dtype"]


def test_keypoint_scale_is_contiguous_and_uncompressed(tmp_path, sample_pose_file):
    """Only contiguous storage makes a frame range one byte range."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r") as h5:
        for component_path in ("/jabs/pose/points", "/jabs/pose/confidence"):
            dataset = h5[component_path]
            assert dataset.chunks is None, f"{component_path} is chunked"
            assert dataset.compression is None, f"{component_path} is compressed"


def test_declared_layout_matches_what_was_written(tmp_path, sample_pose_file):
    """The manifest reports the storage actually used, not an intention."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r") as h5:
        manifest = json.loads(h5["manifest"][()])
        for entry in manifest["components"]:
            dataset = h5[entry["path"]]
            expected = "contiguous" if dataset.chunks is None else "chunked"
            assert entry["layout"]["storage"] == expected
            assert entry["layout"]["compression"] == (dataset.compression or "none")


def test_writer_refuses_an_invalid_file(tmp_path, sample_pose_file):
    """The writer validates what it is given, not just what PoseFile allows.

    The previous version of this test only constructed a PoseFile inside
    pytest.raises and never called write_pose_file, so it asserted the
    writer's safety property without exercising the writer at all -- it would
    have stayed green with validation moved after the truncating open. The
    disk-safety assertions now live in test_writer_safety.py, which writes a
    sentinel and checks it survives.
    """
    points = sample_pose_file.component("jabs.pose.points")
    # A skeleton id PoseFile accepts (the reference resolves) and the schema
    # rejects (ids must be namespaced), so only the writer can catch it.
    unnamespaced = PoseFile(
        dimensions=sample_pose_file.dimensions,
        video=sample_pose_file.video,
        skeletons={"mouse": sample_pose_file.skeletons["jabs.mouse12"]},
        components=(
            Component(
                id=points.id,
                axes=points.axes,
                data=points.data,
                missing=points.missing,
                units=points.units,
                coord_order=points.coord_order,
                skeleton="mouse",
            ),
        ),
        provenance=sample_pose_file.provenance,
    )
    path = tmp_path / "invalid_pose.h5"
    with pytest.raises(ValueError, match="refusing to write an invalid pose file"):
        write_pose_file(unnamespaced, path)
    assert not path.exists()


def test_nan_is_written_not_a_sentinel(tmp_path, sample_pose_file):
    """Missing float coordinates are NaN on disk; sentinels are banned."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r") as h5:
        points = h5["/jabs/pose/points"][()]
    assert np.isnan(points).any()
    assert not (points == 0).all(axis=-1).any(), "a zero coordinate would be ambiguous"
