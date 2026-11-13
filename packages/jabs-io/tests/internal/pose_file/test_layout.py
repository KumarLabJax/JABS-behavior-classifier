"""A component's declared layout is compared in full with the dataset's storage."""

import json

import h5py
import numpy as np

from jabs.io.internal.pose_file import Component, PoseFile, VideoInfo, validate, write_pose_file


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


# --- the declared layout is compared in full ------------------------------


def test_a_wrong_declared_chunk_shape_is_a_warning(tmp_path, sample_pose_file):
    """chunks and compression_opts are part of the layout schema."""
    seg = Component(
        id="jabs.segmentation.contours",
        axes=("frame", "slot"),
        data=np.zeros((4, 2), dtype=np.int32),
        missing={"policy": "none"},
    )
    path = tmp_path / "layout.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
            components=(seg,),
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []
    _rewrite_manifest(
        path,
        lambda m: m["components"][0]["layout"].update(chunks=[999, 999]),
    )
    assert "layout_matches_file" in _checks(path)


def test_a_wrong_declared_compression_level_is_a_warning(tmp_path):
    """A manifest claiming gzip 9 over a level-1 dataset is inaccurate."""
    seg = Component(
        id="jabs.segmentation.contours",
        axes=("frame", "slot"),
        data=np.zeros((4, 2), dtype=np.int32),
        missing={"policy": "none"},
    )
    path = tmp_path / "level.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 4, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=4, width=8, height=8, fps=30.0),
            components=(seg,),
        ),
        path,
    )
    _rewrite_manifest(path, lambda m: m["components"][0]["layout"].update(compression_opts=9))
    assert "layout_matches_file" in _checks(path)
