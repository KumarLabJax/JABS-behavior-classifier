"""The writer validates the finished file before publishing it.

A file the validator rejects is never published; a warning alone does not block a
write.
"""

import numpy as np
import pytest

from jabs.io.internal.pose_file import (
    Component,
    PoseFile,
    VideoInfo,
    validate,
    write_pose_file,
)


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


# --- the writer never publishes what its validator rejects ---------------


def test_the_writer_validates_the_finished_file(tmp_path, sample_pose_file):
    """File-level invariants can only be checked against the written file."""
    index_id = "jabs.dynamic_objects.fecal_boli.frame_index"
    index = Component(
        id=index_id,
        axes=("sample",),
        data=np.array([0, 1, 2], dtype=np.uint32),
        missing={"policy": "none"},
        units="frame",
        sparse_index=index_id,
    )
    path = tmp_path / "ok.h5"
    write_pose_file(
        PoseFile(
            dimensions=sample_pose_file.dimensions,
            video=sample_pose_file.video,
            skeletons=sample_pose_file.skeletons,
            components=(*sample_pose_file.components, index),
            provenance=sample_pose_file.provenance,
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []


def test_a_component_with_no_provenance_is_only_a_warning(tmp_path, sample_pose_file):
    """Requiring it would invalidate a third party's file (design goal 16)."""
    anonymous = Component(
        id="org.example.lab.thing",
        axes=("frame",),
        data=np.zeros(4, dtype=np.float32),
        missing={"policy": "nan"},
        units="unitless",
    )
    path = tmp_path / "anon.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 1, "identity": 1},
            video=VideoInfo(frame_count=4, width=4, height=4, fps=30.0),
            components=(anonymous,),
        ),
        path,
    )
    findings = validate(path)
    assert [f for f in findings if f.severity == "error"] == []
    assert any(f.check == "provenance_declared" for f in findings)
