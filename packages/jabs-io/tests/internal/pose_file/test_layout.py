"""A component's declared layout is compared in full with the dataset's storage."""

import dataclasses
import json
import logging

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file import (
    Component,
    PoseFile,
    VideoInfo,
    read_manifest,
    read_pose_file,
    validate,
    write_pose_file,
)

from .conftest import build_embedding_component, build_sample_pose_file


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


# --- requested layouts and the embedding layout policy ---------------------

_INSTANCE = "jabs.embedding.trunk.instance"
_FOREIGN = "org.example.lab.x"


def _single(component: Component, frames: int, slots: int = 2) -> PoseFile:
    """Wrap one component in a minimal pose file."""
    return PoseFile(
        dimensions={"frame": frames, "slot": slots, "identity": slots},
        video=VideoInfo(frame_count=frames, width=8, height=8, fps=30.0),
        components=(component,),
    )


def _foreign(frames: int = 4, requested_layout: dict | None = None) -> Component:
    """A third-party component under a reverse-DNS namespace."""
    return Component(
        id=_FOREIGN,
        axes=("frame", "slot"),
        data=np.arange(frames * 2, dtype=np.float32).reshape(frames, 2),
        missing={"policy": "none"},
        requested_layout=requested_layout,
    )


def _entry(path, component_id: str) -> dict:
    """Return one component's manifest entry from a written file."""
    return next(c for c in read_manifest(path)["components"] if c["id"] == component_id)


def _write_clean(pose_file: PoseFile, path) -> None:
    """Write a file and assert it validates with no error and no layout warning."""
    write_pose_file(pose_file, path)
    findings = validate(path)
    assert [f for f in findings if f.severity == "error"] == []
    assert "layout_matches_file" not in {f.check for f in findings}


def test_embedding_default_layout_is_chunked_uncompressed(tmp_path):
    """An embedding is chunked along frame, other axes whole, uncompressed."""
    path = tmp_path / "default.h5"
    _write_clean(_single(build_embedding_component(frames=4, slots=2, dim=8), 4), path)
    assert _entry(path, _INSTANCE)["layout"] == {
        "storage": "chunked",
        "chunks": [4, 2, 8],
        "compression": "none",
    }
    with h5py.File(path, "r") as h5:
        dataset = h5["/jabs/embedding/trunk/instance"]
        assert dataset.chunks == (4, 2, 8)
        assert dataset.compression is None


def test_embedding_default_chunk_targets_one_mebibyte(tmp_path):
    """13312 bytes a frame puts 78 frames in a one-MiB chunk."""
    path = tmp_path / "mib.h5"
    component = build_embedding_component(frames=1000, slots=4, dim=1664)
    _write_clean(_single(component, 1000, slots=4), path)
    assert _entry(path, _INSTANCE)["layout"]["chunks"] == [78, 4, 1664]


def test_wide_rows_get_one_frame_chunks(tmp_path):
    """A frame wider than the target still gets a chunk of one frame, never zero."""
    path = tmp_path / "wide.h5"
    component = build_embedding_component(frames=3, slots=8, dim=65536, dtype="float32")
    _write_clean(_single(component, 3, slots=8), path)
    assert _entry(path, _INSTANCE)["layout"]["chunks"] == [1, 8, 65536]


def test_zero_frame_embedding_is_contiguous(tmp_path):
    """An empty payload cannot be chunked, so it is stored contiguous."""
    path = tmp_path / "empty.h5"
    _write_clean(_single(build_embedding_component(frames=0), 0), path)
    assert _entry(path, _INSTANCE)["layout"] == {"storage": "contiguous", "compression": "none"}


def test_requested_layout_is_honored_and_clamped(tmp_path):
    """A request is applied, with chunk extents clamped to the axis lengths."""
    component = dataclasses.replace(
        build_embedding_component(frames=4),
        requested_layout={
            "storage": "chunked",
            "chunks": [500, 2, 8],
            "compression": "gzip",
            "compression_opts": 4,
        },
    )
    path = tmp_path / "requested.h5"
    _write_clean(_single(component, 4), path)
    assert _entry(path, _INSTANCE)["layout"] == {
        "storage": "chunked",
        "chunks": [4, 2, 8],
        "compression": "gzip",
        "compression_opts": 4,
    }
    with h5py.File(path, "r") as h5:
        dataset = h5["/jabs/embedding/trunk/instance"]
        assert dataset.chunks == (4, 2, 8)
        assert (dataset.compression, dataset.compression_opts) == ("gzip", 4)


def test_requested_lzf_layout_is_honored(tmp_path):
    """lzf takes no options, and none are recorded."""
    component = dataclasses.replace(
        build_embedding_component(frames=4),
        requested_layout={"storage": "chunked", "chunks": [2, 2, 8], "compression": "lzf"},
    )
    path = tmp_path / "lzf.h5"
    _write_clean(_single(component, 4), path)
    assert _entry(path, _INSTANCE)["layout"] == {
        "storage": "chunked",
        "chunks": [2, 2, 8],
        "compression": "lzf",
    }


@pytest.mark.parametrize("component_id", ["jabs.pose.points", "jabs.segmentation.contour_count"])
def test_requested_layout_rejected_on_jabs_pose(component_id):
    """The writer owns the layout of every jabs component that is not an embedding."""
    with pytest.raises(ValueError, match="requested_layout"):
        Component(
            id=component_id,
            axes=("frame", "slot"),
            data=np.zeros((4, 2), dtype=np.int32),
            missing={"policy": "none"},
            requested_layout={"storage": "contiguous", "compression": "none"},
        )


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param({"storage": "contiguous", "compression": "gzip"}, id="contiguous-gzip"),
        pytest.param(
            {"storage": "chunked", "chunks": [4, 2, 8], "compression": "szip"}, id="szip"
        ),
        pytest.param(
            {
                "storage": "chunked",
                "chunks": [4, 2, 8],
                "compression": "gzip",
                "compression_opts": 10,
            },
            id="gzip-10",
        ),
        pytest.param(
            {
                "storage": "chunked",
                "chunks": [4, 2, 8],
                "compression": "lzf",
                "compression_opts": 1,
            },
            id="lzf-opts",
        ),
        pytest.param(
            {"storage": "chunked", "chunks": [4, 2], "compression": "none"}, id="chunk-rank"
        ),
        pytest.param(
            {"storage": "chunked", "chunks": [0, 2, 8], "compression": "none"}, id="chunk-zero"
        ),
        pytest.param(
            {
                "storage": "chunked",
                "chunks": [4, 2, 8],
                "compression": "gzip",
                "compression_opts": True,
            },
            id="gzip-bool",
        ),
        pytest.param(
            {"storage": "chunked", "chunks": [4, 2, 8], "compression": "gzip"},
            id="gzip-no-level",
        ),
        pytest.param({"storage": "chunked", "compression": "none"}, id="chunked-no-chunks"),
        pytest.param(
            {"storage": "contiguous", "chunks": [4, 2, 8], "compression": "none"},
            id="contiguous-chunks",
        ),
        pytest.param({"storage": "contiguous", "shuffle": True}, id="unknown-key"),
        pytest.param({"storage": "virtual", "compression": "none"}, id="storage"),
    ],
)
def test_requested_layout_validation(layout):
    """An unsatisfiable or out-of-policy request fails at construction."""
    with pytest.raises(ValueError, match="requested_layout"):
        dataclasses.replace(build_embedding_component(frames=4), requested_layout=layout)


def test_round_trip_keeps_embedding_and_foreign_layouts(tmp_path):
    """Read then rewrite keeps the requested layouts; jabs.pose stays the writer's."""
    base = build_sample_pose_file()
    embedding = dataclasses.replace(
        build_embedding_component(frames=4),
        requested_layout={
            "storage": "chunked",
            "chunks": [2, 2, 8],
            "compression": "gzip",
            "compression_opts": 4,
        },
    )
    foreign = _foreign(requested_layout={"storage": "contiguous", "compression": "none"})
    first = tmp_path / "first.h5"
    _write_clean(
        dataclasses.replace(base, components=(*base.components, embedding, foreign)), first
    )

    second = tmp_path / "second.h5"
    _write_clean(read_pose_file(first), second)
    for component_id in (_INSTANCE, _FOREIGN):
        assert _entry(second, component_id)["layout"] == _entry(first, component_id)["layout"]
    assert _entry(second, _INSTANCE)["layout"]["compression_opts"] == 4
    assert _entry(second, _FOREIGN)["layout"] == {"storage": "contiguous", "compression": "none"}
    assert _entry(second, "jabs.pose.points")["layout"] == {
        "storage": "contiguous",
        "compression": "none",
    }


def test_foreign_without_request_keeps_default_policy(tmp_path):
    """A foreign component with no request gets the writer's default chunked gzip policy."""
    path = tmp_path / "foreign.h5"
    _write_clean(_single(_foreign(frames=600), 600), path)
    assert _entry(path, _FOREIGN)["layout"] == {
        "storage": "chunked",
        "chunks": [600, 2],
        "compression": "gzip",
        "compression_opts": 4,
    }


def test_unusable_foreign_layout_reads_as_no_request(tmp_path, caplog):
    """A foreign layout the writer cannot honor is not a request, and never fails a read."""
    path = tmp_path / "szip.h5"
    write_pose_file(_single(_foreign(), 4), path)
    _rewrite_manifest(
        path,
        lambda m: m["components"][0]["layout"].update(compression="szip"),
    )
    with caplog.at_level(logging.DEBUG, logger="jabs.io.internal.pose_file.reader"):
        component = read_pose_file(path).component(_FOREIGN)
    assert component.requested_layout is None
    assert component.layout["compression"] == "szip"
    assert any(_FOREIGN in r.getMessage() and "szip" in r.getMessage() for r in caplog.records)


def test_reader_does_not_request_layouts_for_jabs_components(tmp_path, sample_pose_file):
    """A jabs.pose component's recorded layout is the writer's, not a request."""
    path = tmp_path / "pose.h5"
    write_pose_file(sample_pose_file, path)
    assert read_pose_file(path).component("jabs.pose.points").requested_layout is None
