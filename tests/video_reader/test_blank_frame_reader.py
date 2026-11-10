"""Tests for BlankFrameReader."""

import numpy as np
import pytest

from jabs.video_reader import BlankFrameReader


def test_frames_are_solid_color_at_the_requested_size() -> None:
    """Every frame is the requested color, in OpenCV's (height, width, BGR) layout."""
    reader = BlankFrameReader(num_frames=3, fps=30, width=40, height=20, color=(1, 2, 3))

    frame = reader.load_next_frame()

    assert frame["data"].shape == (20, 40, 3)
    assert frame["data"].dtype == np.uint8
    assert (frame["data"] == np.array([1, 2, 3], dtype=np.uint8)).all()
    assert frame["index"] == 0
    assert frame["duration"] == pytest.approx(1 / 30)
    assert reader.dimensions == (40, 20)


def test_frames_advance_and_end_with_eof() -> None:
    """Frames are numbered in order and the reader signals EOF after the last one."""
    reader = BlankFrameReader(num_frames=2, fps=30, width=4, height=4)

    assert reader.load_next_frame()["index"] == 0
    assert reader.load_next_frame()["index"] == 1
    eof = reader.load_next_frame()
    assert eof["data"] is None
    assert eof["index"] == -1


def test_seek_moves_to_the_requested_frame() -> None:
    """Seeking sets which frame is read next."""
    reader = BlankFrameReader(num_frames=10, fps=30, width=4, height=4)

    reader.seek(7)

    assert reader.load_next_frame()["index"] == 7


def test_returned_frames_do_not_share_memory() -> None:
    """A caller drawing into one frame must not change the next."""
    reader = BlankFrameReader(num_frames=2, fps=30, width=4, height=4)

    first = reader.load_next_frame()["data"]
    first[:] = 255

    assert (reader.load_next_frame()["data"] == 0).all()


def test_frame_time_uses_the_frame_rate() -> None:
    """Frame times are formatted the same way VideoReader formats them."""
    reader = BlankFrameReader(num_frames=10_000, fps=30, width=4, height=4)

    assert reader.get_frame_time(30 * 61) == "00:01:01"


@pytest.mark.parametrize(
    ("num_frames", "fps", "width", "height"),
    [(10, 0, 4, 4), (10, 30, 0, 4), (10, 30, 4, 0), (-1, 30, 4, 4)],
    ids=["zero-fps", "zero-width", "zero-height", "negative-frames"],
)
def test_invalid_arguments_raise(num_frames: int, fps: int, width: int, height: int) -> None:
    """Sizes and rates that cannot be played are rejected up front."""
    with pytest.raises(ValueError):
        BlankFrameReader(num_frames=num_frames, fps=fps, width=width, height=height)
