"""The writer streams a block-sourced component to disk a block at a time."""

import dataclasses
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file import validate, write_pose_file
from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.types import Component, PoseFile, VideoInfo

from .conftest import build_sample_pose_file

CREATED = "2026-11-18T00:00:00Z"
# A compressed (non-keypoint-scale) component, chunked and gzipped by the writer.
COMPRESSED_ID = "org.example.lab.whisker_angle"
# A keypoint-scale component, contiguous and uncompressed.
KEYPOINT_ID = "jabs.pose.points"
SENTINEL = b"the previous file"

Blocks = Iterable[tuple[int, npt.NDArray[np.generic]]]


def _as_blocks(array: npt.NDArray[np.generic], rows: int) -> BlockSource:
    """A block source yielding consecutive slices of ``rows`` rows."""

    def blocks() -> Iterator[tuple[int, npt.NDArray[np.generic]]]:
        for start in range(0, array.shape[0], rows):
            yield start, array[start : start + rows]

    return BlockSource(array.shape, array.dtype, blocks)


def _in_memory() -> PoseFile:
    """The sample file plus a compressed foreign component, all in memory."""
    sample = build_sample_pose_file()
    angle = np.random.default_rng(11).uniform(-1, 1, (4, 2)).astype(np.float32)
    whisker = Component(
        id=COMPRESSED_ID,
        axes=("frame", "slot"),
        data=angle,
        missing={"policy": "nan"},
        units="radian",
    )
    return dataclasses.replace(sample, components=(*sample.components, whisker))


def _with_sources(pose_file: PoseFile, sources: dict[str, BlockSource]) -> PoseFile:
    """The same file with the named components' data swapped for block sources."""
    components = tuple(
        dataclasses.replace(c, data=sources[c.id]) if c.id in sources else c
        for c in pose_file.components
    )
    return dataclasses.replace(pose_file, components=components)


def _streamed(pose_file: PoseFile) -> PoseFile:
    """The file with one compressed and one keypoint-scale component streamed."""
    return _with_sources(
        pose_file,
        {
            component_id: _as_blocks(pose_file.component(component_id).data, rows=3)
            for component_id in (COMPRESSED_ID, KEYPOINT_ID)
        },
    )


def _datasets(path: Path) -> dict[str, dict]:
    """Every dataset's values and storage options, keyed by path."""
    found: dict[str, dict] = {}

    def visit(name: str, node: object) -> None:
        if isinstance(node, h5py.Dataset):
            found[name] = {
                "value": node[()],
                "chunks": node.chunks,
                "compression": node.compression,
                "compression_opts": node.compression_opts,
                "shuffle": node.shuffle,
            }

    with h5py.File(path, "r") as h5:
        h5.visititems(visit)
    return found


def test_streamed_and_in_memory_writes_match(tmp_path: Path) -> None:
    """Streaming changes how the payload arrives, not what is written."""
    pose_file = _in_memory()
    memory_path, streamed_path = tmp_path / "memory.h5", tmp_path / "streamed.h5"
    write_pose_file(pose_file, memory_path, created=CREATED)
    write_pose_file(_streamed(pose_file), streamed_path, created=CREATED)

    expected, actual = _datasets(memory_path), _datasets(streamed_path)
    assert expected.keys() == actual.keys()
    assert expected["org.example.lab/whisker_angle"]["compression"] == "gzip"
    for name, want in expected.items():
        got = actual[name]
        if isinstance(want["value"], bytes):
            assert got["value"] == want["value"], name
        else:
            np.testing.assert_array_equal(got["value"], want["value"], err_msg=name)
        for option in ("chunks", "compression", "compression_opts", "shuffle"):
            assert got[option] == want[option], (name, option)


def test_blocks_is_called_exactly_once(tmp_path: Path) -> None:
    """Validation and manifest-building read only shape and dtype."""
    pose_file = _in_memory()
    array = pose_file.component(COMPRESSED_ID).data
    inner = _as_blocks(array, rows=3)
    calls = 0

    def counted() -> Blocks:
        nonlocal calls
        calls += 1
        return inner.blocks()

    source = BlockSource(array.shape, array.dtype, counted)
    write_pose_file(_with_sources(pose_file, {COMPRESSED_ID: source}), tmp_path / "once.h5")
    assert calls == 1


def test_a_zero_frame_block_source_writes(tmp_path: Path) -> None:
    """An empty source yields nothing and still writes a valid file."""
    empty = Component(
        id="org.example.lab.empty",
        axes=("frame", "slot"),
        data=BlockSource((0, 2), "float32", lambda: iter(())),
        missing={"policy": "nan"},
        units="unitless",
    )
    path = tmp_path / "empty.h5"
    write_pose_file(
        PoseFile(
            dimensions={"frame": 0, "slot": 2, "identity": 2},
            video=VideoInfo(frame_count=0, width=8, height=8, fps=30.0),
            components=(empty,),
        ),
        path,
    )
    assert [f for f in validate(path) if f.severity == "error"] == []
    with h5py.File(path, "r") as h5:
        dataset = h5["org.example.lab/empty"]
        assert dataset.shape == (0, 2)
        assert dataset.dtype == np.float32
        assert dataset.chunks is None


def _overlap(array: npt.NDArray[np.float32]) -> Blocks:
    """The second block restarts at row 0."""
    return [(0, array[:2]), (0, array[2:])]


def _gap(array: npt.NDArray[np.float32]) -> Blocks:
    """The second block skips row 2."""
    return [(0, array[:2]), (3, array[2:])]


def _short(array: npt.NDArray[np.float32]) -> Blocks:
    """The blocks stop one row early."""
    return [(0, array[:3])]


def _overrun(array: npt.NDArray[np.float32]) -> Blocks:
    """The last block passes the declared first-axis length."""
    return [(0, array), (4, array[:1])]


def _wrong_trailing_shape(array: npt.NDArray[np.float32]) -> Blocks:
    """A block whose trailing axes disagree with the declared shape."""
    return [(0, np.zeros((4, 3), dtype=np.float32))]


def _wrong_dtype(array: npt.NDArray[np.float32]) -> Blocks:
    """float64 rows offered to a float32 declaration."""
    return [(0, array.astype(np.float64))]


@pytest.mark.parametrize(
    "bad_blocks",
    [_overlap, _gap, _short, _overrun, _wrong_trailing_shape, _wrong_dtype],
    ids=["overlap", "gap", "short", "overrun", "trailing-shape", "dtype"],
)
def test_bad_blocks_are_refused(
    tmp_path: Path, bad_blocks: Callable[[npt.NDArray[np.float32]], Blocks]
) -> None:
    """A source that does not cover its declared shape exactly fails the write."""
    pose_file = _in_memory()
    array = pose_file.component(COMPRESSED_ID).data
    source = BlockSource(array.shape, array.dtype, lambda: bad_blocks(array))
    path = tmp_path / "out.h5"
    path.write_bytes(SENTINEL)

    with pytest.raises(ValueError, match=COMPRESSED_ID):
        write_pose_file(_with_sources(pose_file, {COMPRESSED_ID: source}), path)

    assert path.read_bytes() == SENTINEL
    assert list(tmp_path.glob("*.tmp")) == []
