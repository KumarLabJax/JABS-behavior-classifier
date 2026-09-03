"""Tests for writing pose files."""

import json

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.schema import validate_manifest, validate_provenance
from jabs.io.internal.pose_file.types import PoseFile, VideoInfo
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


def test_invalid_pose_file_does_not_touch_disk(tmp_path, sample_pose_file):
    """h5py truncates at open, so validation must happen before opening."""
    path = tmp_path / "a_pose.h5"
    path.write_bytes(b"sentinel")
    with pytest.raises(ValueError, match="identity"):
        PoseFile(
            dimensions={"frame": 4, "slot": 1, "identity": 5},
            video=VideoInfo(frame_count=4),
        )
    assert path.read_bytes() == b"sentinel"


def test_nan_is_written_not_a_sentinel(tmp_path, sample_pose_file):
    """Missing float coordinates are NaN on disk; sentinels are banned."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r") as h5:
        points = h5["/jabs/pose/points"][()]
    assert np.isnan(points).any()
    assert not (points == 0).all(axis=-1).any(), "a zero coordinate would be ambiguous"
