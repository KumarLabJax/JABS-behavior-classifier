"""Files written by other HDF5 writers.

Attribute types h5py does not produce itself are reported, not recursed into or
crashed on.
"""

import sys

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file import (
    PoseFileError,
    read_pose_file,
    validate,
    write_pose_file,
)
from jabs.io.internal.pose_file.reader import attr_text


def _checks(path):
    return {f.check for f in validate(path)}


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


# --- interop with other HDF5 writers --------------------------------------


def test_a_numeric_format_attribute_is_rejected_not_recursed(tmp_path, sample_pose_file):
    """attr_text used to recurse forever on a non-string scalar."""
    assert attr_text(1) is None
    assert attr_text(np.int32(7)) is None
    path = tmp_path / "numeric.h5"
    write_pose_file(sample_pose_file, path)
    with h5py.File(path, "r+") as h5:
        del h5.attrs["jabs_format"]
        h5.attrs["jabs_format"] = 1
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        assert "root_attrs" in _checks(path)
        with pytest.raises(PoseFileError):
            read_pose_file(path)
    finally:
        sys.setrecursionlimit(limit)
