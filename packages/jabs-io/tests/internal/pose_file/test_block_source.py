"""A block source stands in for an in-memory array as component data."""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pytest

from jabs.io.internal.pose_file.blocks import BlockSource
from jabs.io.internal.pose_file.types import Component, PoseFile, VideoInfo


def _no_blocks() -> Iterable[tuple[int, npt.NDArray[np.generic]]]:
    """Yield nothing."""
    return iter(())


def _never_read() -> Iterable[tuple[int, npt.NDArray[np.generic]]]:
    """Fail the test if construction tries to read the blocks."""
    raise AssertionError("construction read the blocks of a block source")


def test_a_block_source_declares_shape_dtype_ndim_and_size() -> None:
    """Shape and dtype are declared up front, so ndim and size need no data."""
    source = BlockSource((10, 3), "float32", _no_blocks)
    assert source.ndim == 2
    assert source.size == 30
    assert source.dtype == np.dtype("float32")


@pytest.mark.parametrize("dtype", ["O", "U4"])
def test_a_string_block_source_is_refused(dtype: str) -> None:
    """String components stay in memory; a block source cannot carry them."""
    with pytest.raises(ValueError, match="string"):
        BlockSource((4,), dtype, _no_blocks)


def test_a_component_accepts_a_block_source() -> None:
    """A component built over a block source reports the declared dtype."""
    component = Component(
        id="jabs.pose.slot_occupied",
        axes=("frame", "slot"),
        data=BlockSource((5, 2), "bool", _no_blocks),
        missing={"policy": "none"},
    )
    assert component.dtype == "bool"


def test_axes_must_match_a_block_source_rank() -> None:
    """Three axes cannot label a two-dimensional source."""
    with pytest.raises(ValueError, match="axes"):
        Component(
            id="jabs.pose.slot_occupied",
            axes=("frame", "slot", "keypoint"),
            data=BlockSource((5, 2), "bool", _no_blocks),
            missing={"policy": "none"},
        )


def test_a_length_reference_from_a_block_source_is_not_read() -> None:
    """Constructing a file never reads a block-source length reference."""
    lengths = Component(
        id="jabs.test.lengths",
        axes=("frame", "slot"),
        data=BlockSource((5, 2), "uint32", _never_read),
        missing={"policy": "none"},
    )
    values = Component(
        id="jabs.test.values",
        axes=("frame", "slot", "sample_point"),
        data=np.zeros((5, 2, 4), dtype=np.float32),
        missing={"policy": "length", "length": "jabs.test.lengths"},
    )
    pose = PoseFile(
        dimensions={"frame": 5, "slot": 2, "identity": 2},
        video=VideoInfo(frame_count=5),
        components=(lengths, values),
    )
    assert pose.component("jabs.test.lengths").data is lengths.data
