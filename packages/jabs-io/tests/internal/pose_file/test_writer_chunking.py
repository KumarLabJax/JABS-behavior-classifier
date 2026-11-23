"""Chunked components are chunked per item run and shuffled, and chunks stay small."""

import dataclasses
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.io.internal.pose_file import validate, write_pose_file
from jabs.io.internal.pose_file.types import Component
from jabs.io.internal.pose_file.writer import CHUNK_BYTES, COMPRESSION_LEVEL, chunk_shape

from .conftest import build_sample_pose_file

_CONTOUR_AXES = ("frame", "slot", "contour", "point", "coord")


@pytest.mark.parametrize(
    ("axes", "shape", "itemsize", "kept"),
    [
        # One contour's x or y: the point axis whole, coord split.
        (_CONTOUR_AXES, (1800, 3, 5, 1108, 2), 4, 3),
        (_CONTOUR_AXES, (1800, 10, 79, 5923, 2), 4, 3),
        # Without a coord axis, the innermost axis is the item axis.
        (("frame", "slot", "contour"), (1800, 3, 5), 4, 2),
        (("frame", "slot"), (1800, 3), 1, 1),
        # A coord axis alone is still kept.
        (("frame", "coord"), (1800, 2), 4, 1),
    ],
)
def test_a_chunk_is_frames_of_one_item_run(axes, shape, itemsize, kept) -> None:
    """Every trailing axis but the item axis is 1, and frames fill the chunk's budget."""
    chunks = chunk_shape(axes, shape, itemsize)
    for axis in range(1, len(shape)):
        assert chunks[axis] == (shape[axis] if axis == kept else 1), (axis, chunks)
    row = math.prod(chunks[1:]) * itemsize
    assert chunks[0] == min(shape[0], max(1, CHUNK_BYTES // row))
    assert math.prod(chunks) * itemsize <= max(CHUNK_BYTES, row)


def test_the_frame_extent_does_not_depend_on_the_slot_count() -> None:
    """The converter sizes its blocks before it knows the slot count."""
    rows = {chunk_shape(_CONTOUR_AXES, (1800, s, 79, 5923, 2), 4)[0] for s in (1, 5, 10, 64)}
    assert len(rows) == 1


def test_a_huge_item_axis_is_split_to_the_budget() -> None:
    """No chunk passes the budget, so none approaches HDF5 1.x's 4 GiB chunk limit."""
    chunks = chunk_shape(("frame", "sample"), (10, 2**31), 4)
    assert chunks == (1, CHUNK_BYTES // 4)


def test_a_chunked_component_is_written_with_its_chunks_shuffled(tmp_path: Path) -> None:
    """Contours get item-run chunks, gzip and shuffle; the manifest records the layout."""
    rng = np.random.default_rng(3)
    sample = build_sample_pose_file()
    frames, slots = sample.dimensions["frame"], sample.dimensions["slot"]
    contours = rng.integers(-1, 500, (frames, slots, 3, 40, 2), dtype=np.int32)
    lengths = np.full((frames, slots, 3), 40, dtype=np.uint32)
    seg = (
        Component(
            id="jabs.segmentation.contours",
            axes=_CONTOUR_AXES,
            data=contours,
            missing={"policy": "length", "length": "jabs.segmentation.contour_length"},
            units="pixel",
            coord_order="xy",
        ),
        Component(
            id="jabs.segmentation.contour_length",
            axes=("frame", "slot", "contour"),
            data=lengths,
            missing={"policy": "none"},
        ),
    )
    path = tmp_path / "seg.h5"
    write_pose_file(dataclasses.replace(sample, components=(*sample.components, *seg)), path)

    assert [f for f in validate(path) if f.severity == "error"] == []
    with h5py.File(path, "r") as h5:
        layouts = {c["id"]: c["layout"] for c in json.loads(h5["manifest"][()])["components"]}
        for component in seg:
            dataset = h5[component.path]
            want = chunk_shape(component.axes, component.data.shape, component.data.itemsize)
            assert dataset.chunks == want, component.id
            assert dataset.compression == "gzip", component.id
            assert dataset.compression_opts == COMPRESSION_LEVEL, component.id
            assert dataset.shuffle, component.id
            assert layouts[component.id] == {
                "storage": "chunked",
                "chunks": list(want),
                "compression": "gzip",
                "compression_opts": COMPRESSION_LEVEL,
            }
            np.testing.assert_array_equal(dataset[...], component.data)
