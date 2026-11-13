"""The public ``jabs.io`` save path forwards its options to the pose writer."""

import json

import h5py
import pytest

import jabs.io
from jabs.io.internal.pose_file import (
    write_pose_file,
)


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


# --- the public API forwards its options ---------------------------------


def test_public_save_forwards_created(tmp_path, sample_pose_file):
    """jabs.io.save discarded kwargs, so `created` silently used now."""
    path = tmp_path / "stamped.h5"
    jabs.io.save(sample_pose_file, path, created="2026-09-08T00:00:00Z")
    with h5py.File(path, "r") as h5:
        assert json.loads(h5["manifest"][()])["created"] == "2026-09-08T00:00:00Z"


def test_public_save_rejects_an_unsupported_option(tmp_path, sample_pose_file):
    """An unsupported option must fail visibly rather than be dropped."""
    with pytest.raises(TypeError):
        jabs.io.save(sample_pose_file, tmp_path / "x.h5", nonsense=True)
