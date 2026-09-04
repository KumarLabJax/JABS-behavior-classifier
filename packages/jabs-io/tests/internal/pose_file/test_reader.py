"""Tests for reading pose files."""

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file.reader import (
    NotAPoseFileError,
    read_component,
    read_manifest,
    read_pose_file,
)
from jabs.io.internal.pose_file.writer import write_pose_file


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


def test_round_trip_preserves_arrays(written, sample_pose_file):
    """Every payload survives the round trip bit for bit."""
    back = read_pose_file(written)
    for original in sample_pose_file.components:
        np.testing.assert_array_equal(
            back.component(original.id).data,
            original.data,
            err_msg=f"{original.id} changed in the round trip",
        )


def test_round_trip_preserves_declarations(written, sample_pose_file):
    """And so do the declarations that make the payload interpretable."""
    back = read_pose_file(written)
    points = back.component("jabs.pose.points")
    assert points.axes == ("frame", "slot", "keypoint", "coord")
    assert points.units == "pixel"
    assert points.coord_order == "xy"
    assert points.missing == {"policy": "nan"}
    assert points.skeleton == "jabs.mouse12"
    assert points.provenance == "jabs.pose"
    assert back.dimensions == sample_pose_file.dimensions
    assert back.video.fps == 30.0
    assert back.video.width == 800
    assert back.skeletons["jabs.mouse12"].edges == sample_pose_file.skeletons["jabs.mouse12"].edges


def test_round_trip_preserves_provenance(written):
    """Provenance comes back typed, including the declared policy."""
    back = read_pose_file(written)
    record = back.provenance.records["jabs.pose"]
    assert record.producer == "test"
    assert record.parameters == {"confidence_threshold": 0.3}
    assert back.provenance.history[0].operation == "infer"


def test_nan_survives_the_round_trip(written):
    """Masked keypoints stay NaN rather than becoming a coordinate."""
    points = read_component(written, "jabs.pose.points")
    assert np.isnan(points).any()


def test_partial_read_matches_the_full_read(written):
    """A frame window is a slice of the whole, and reads only that slice."""
    full = read_component(written, "jabs.pose.points")
    window = read_component(written, "jabs.pose.points", frames=slice(1, 3))
    np.testing.assert_array_equal(window, full[1:3])
    assert window.shape[0] == 2


def test_partial_read_rejects_a_component_with_no_frame_axis(tmp_path, sample_pose_file):
    """Slicing a frame range out of something with no frame axis is a mistake."""
    from jabs.io.internal.pose_file.types import Component, PoseFile

    centers = Component(
        id="jabs.identity.centers",
        axes=("identity", "embedding"),
        data=np.zeros((2, 1), dtype=np.float32),
        missing={"policy": "none"},
    )
    path = tmp_path / "b_pose.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=(*sample_pose_file.components, centers),
            provenance=sample_pose_file.provenance,
        ),
        path,
    )
    with pytest.raises(ValueError, match="frame"):
        read_component(path, "jabs.identity.centers", frames=slice(0, 1))


def test_unknown_component_raises_key_error(written):
    """An unknown id is a lookup failure, not a file problem."""
    with pytest.raises(KeyError):
        read_component(written, "jabs.pose.nonexistent")


def test_legacy_file_is_refused_clearly(tmp_path):
    """A legacy file gets a message naming what it actually is."""
    legacy = tmp_path / "vid_pose_est_v6.h5"
    with h5py.File(legacy, "w") as h5:
        group = h5.create_group("poseest")
        group.attrs["version"] = np.array([6, 0], dtype=np.uint16)
        group.create_dataset("points", data=np.zeros((2, 1, 12, 2), dtype=np.uint16))
    with pytest.raises(NotAPoseFileError, match="pose_est_v6"):
        read_pose_file(legacy)


def test_unrelated_hdf5_file_is_refused(tmp_path):
    """So does an HDF5 file that is not a pose file at all."""
    other = tmp_path / "features.h5"
    with h5py.File(other, "w") as h5:
        h5.create_dataset("something", data=np.zeros(3))
    with pytest.raises(NotAPoseFileError):
        read_pose_file(other)


def test_read_manifest_does_not_load_arrays(written):
    """The contents are discoverable without touching a payload."""
    manifest = read_manifest(written)
    assert manifest["format"] == "jabs.pose-file"
    assert {c["id"] for c in manifest["components"]} == {
        "jabs.pose.points",
        "jabs.pose.confidence",
        "jabs.pose.point_valid",
        "jabs.pose.slot_occupied",
    }
