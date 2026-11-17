"""String payloads keep their shape, dtype and representation.

Through a round trip and through a windowed read.
"""

import numpy as np
import pytest

from jabs.io.internal.pose_file import (
    Component,
    PoseFile,
    VideoInfo,
    read_component,
    read_pose_file,
    validate,
    write_pose_file,
)


@pytest.fixture
def written(tmp_path, sample_pose_file):
    """A written copy of the sample pose file."""
    path = tmp_path / "a_pose.h5"
    write_pose_file(sample_pose_file, path)
    return path


# --- string payloads keep their shape, dtype and representation ----------


def test_a_two_dimensional_string_component_round_trips(tmp_path, sample_pose_file):
    """`[str(v) for v in arr.tolist()]` stringified whole rows into one element."""
    names = np.array([["a", "b"], ["c", "d"]], dtype=object)
    component = Component(
        id="org.example.lab.labels",
        axes=("identity", "embedding"),
        data=names,
        missing={"policy": "none"},
    )
    path = tmp_path / "labels.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 1, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=1, width=4, height=4, fps=30.0),
            components=(component,),
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []
    back = read_pose_file(path).component("org.example.lab.labels")
    assert back.data.shape == (2, 2)
    assert back.data[1][0] == "c"


def test_byte_strings_are_decoded_not_repred(tmp_path):
    """b"a" became the literal text "b'a'"."""
    component = Component(
        id="org.example.lab.labels",
        axes=("identity",),
        data=np.array([b"a", b"b"], dtype=object),
        missing={"policy": "none"},
    )
    path = tmp_path / "bytes.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 1, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=1, width=4, height=4, fps=30.0),
            components=(component,),
        ),
        path,
    )
    assert list(read_component(path, "org.example.lab.labels")) == ["a", "b"]


def test_a_string_window_matches_a_whole_read(tmp_path):
    """The representation must not depend on whether frames was supplied."""
    component = Component(
        id="org.example.lab.notes",
        axes=("frame",),
        data=np.array(["w", "x", "y", "z"], dtype=object),
        missing={"policy": "none"},
    )
    path = tmp_path / "notes.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 1, "identity": 1},
            video=VideoInfo(frame_count=4, width=4, height=4, fps=30.0),
            components=(component,),
        ),
        path,
    )
    whole = read_component(path, "org.example.lab.notes")
    window = read_component(path, "org.example.lab.notes", frames=slice(1, 3))
    assert list(window) == list(whole[1:3]) == ["x", "y"]
