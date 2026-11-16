"""Tests for VideoReader, in particular that seek() lands on the requested frame."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from jabs.video_reader import VideoReader
from jabs.video_reader import video_reader as video_reader_module

_FPS = 30
_FRAME_MS = 1000.0 / _FPS


class FakeCapture:
    """Stand-in for cv2.VideoCapture that seeks the way OpenCV does on some files.

    Every frame is evenly spaced at 30 fps, but the capture reports a slightly higher
    frame rate - the frame count over a duration one frame interval short, as in the
    recordings where this bug was found. Seeking converts the frame number to a time
    with that reported rate, so a seek in the second half of the file lands one frame
    early. ``landing_shift`` moves every landing further, to test overshooting.
    Seeking to frame 0 always lands on the first frame, as OpenCV's does.
    ``reported_count`` stands in for a container whose frame count is only an estimate.

    Each frame's pixels hold its true frame number, so a test can tell which frame it
    actually received.
    """

    def __init__(self, num_frames: int, landing_shift: int = 0, reported_count: int | None = None):
        self.num_frames = num_frames
        self.landing_shift = landing_shift
        self.reported_count = reported_count or num_frames
        self.reported_fps = _FPS * num_frames / (num_frames - 1)
        self._next = 0  # frame the next grab() decodes
        self._grabbed: int | None = None  # frame the last grab() decoded

    def isOpened(self) -> bool:
        """Report the capture as open."""
        return True

    def get(self, prop: int) -> float:
        """Return the properties VideoReader reads."""
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.reported_count)
        if prop == cv2.CAP_PROP_FPS:
            return self.reported_fps
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            return 4.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            # like OpenCV, the true timestamp of the last decoded frame
            return 0.0 if self._grabbed is None else self._grabbed * _FRAME_MS
        raise AssertionError(f"unexpected property {prop}")

    def set(self, prop: int, value: float) -> bool:
        """Seek the way OpenCV does: by time, using the reported frame rate."""
        assert prop == cv2.CAP_PROP_POS_FRAMES
        landing = 0 if value == 0 else round(value * _FPS / self.reported_fps) + self.landing_shift
        self._next = min(max(landing, 0), self.num_frames)
        self._grabbed = None
        return True

    def grab(self) -> bool:
        """Decode the next frame."""
        if self._next >= self.num_frames:
            return False
        self._grabbed = self._next
        self._next += 1
        return True

    def retrieve(self) -> tuple[bool, npt.NDArray[np.uint8] | None]:
        """Return the last decoded frame."""
        if self._grabbed is None:
            return False, None
        return True, _frame_holding(self._grabbed)

    def read(self) -> tuple[bool, npt.NDArray[np.uint8] | None]:
        """Decode and return the next frame."""
        return self.retrieve() if self.grab() else (False, None)

    def release(self) -> None:
        """Nothing to release."""


def _frame_holding(number: int) -> npt.NDArray[np.uint8]:
    """A tiny frame whose pixels encode ``number`` (base 256, little endian)."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[0, 0] = (number % 256, number // 256, 0)
    return frame


def _number_in(frame: npt.NDArray[np.uint8]) -> int:
    """Read back the number encoded by ``_frame_holding``."""
    return int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])


def _open_fake(monkeypatch, tmp_path: Path, capture: FakeCapture) -> VideoReader:
    """Open a VideoReader whose OpenCV capture is ``capture``."""
    monkeypatch.setattr(video_reader_module.cv2, "VideoCapture", lambda _path: capture)
    return VideoReader(tmp_path / "video.mp4")


def _assert_plays_in_sync(reader: VideoReader, start: int, frames: int = 3) -> None:
    """Each frame read from ``start`` on is labeled with the frame it really is."""
    for expected in range(start, start + frames):
        frame = reader.load_next_frame()
        assert frame["index"] == expected
        assert _number_in(frame["data"]) == expected, f"labeled {expected}"


def test_seek_corrects_opencv_landing_a_frame_early(monkeypatch, tmp_path) -> None:
    """Seeks into the half of the file where OpenCV lands early still reach the frame."""
    capture = FakeCapture(num_frames=100)
    reader = _open_fake(monkeypatch, tmp_path, capture)

    # confirm the fake reproduces the bug before trusting the result
    capture.set(cv2.CAP_PROP_POS_FRAMES, 80)
    capture.grab()
    assert capture._grabbed == 79

    for target in range(97):
        reader.seek(target)
        _assert_plays_in_sync(reader, target)


def test_seek_corrects_a_short_video(monkeypatch, tmp_path) -> None:
    """Short videos, where the frame rate is off by more, are corrected too."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=20))

    for target in range(17):
        reader.seek(target)
        _assert_plays_in_sync(reader, target)


def test_seek_backs_up_after_overshooting(monkeypatch, tmp_path) -> None:
    """A landing past the requested frame is retried further back."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100, landing_shift=2))

    for target in (0, 1, 30, 75, 96):
        reader.seek(target)
        _assert_plays_in_sync(reader, target)


def test_seek_past_the_end_reports_end_of_file(monkeypatch, tmp_path) -> None:
    """Seeking beyond the last frame leaves the next read at end of file."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100))

    reader.seek(150)

    assert reader.load_next_frame()["data"] is None


def test_seek_trusts_opencv_when_timing_disagrees_with_frame_rate(
    monkeypatch, tmp_path, caplog
) -> None:
    """Without trustworthy timing, seek() falls back to OpenCV's position and says so.

    Here the container's frame count is only an estimate (90 for a 100-frame video), so
    the measured frame duration disagrees with the frame rate and can't be trusted.
    """
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100, reported_count=90))

    with caplog.at_level(logging.WARNING, logger=video_reader_module.__name__):
        reader.seek(10)

    assert reader.load_next_frame()["index"] == 10
    assert "disagrees with its frame rate" in caplog.text


def test_reading_without_seeking_is_unchanged(monkeypatch, tmp_path) -> None:
    """Sequential reads from the start label every frame correctly."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100))

    _assert_plays_in_sync(reader, 0, frames=100)
    assert reader.load_next_frame()["data"] is None


def _write_numbered_video(path: Path, num_frames: int) -> None:
    """Write an MJPEG video whose frames show their frame number as black/white blocks."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), _FPS, (160, 64))
    assert writer.isOpened(), "MJPEG writer unavailable"
    for i in range(num_frames):
        frame = np.zeros((64, 160, 3), dtype=np.uint8)
        for bit in range(8):
            if (i >> bit) & 1:
                frame[:, bit * 20 : bit * 20 + 20] = 255
        writer.write(frame)
    writer.release()


def _decode_number(frame: npt.NDArray[np.uint8]) -> int:
    """Read back the number drawn by ``_write_numbered_video``."""
    return sum(int(frame[:, bit * 20 + 5 : bit * 20 + 15].mean() > 127) << bit for bit in range(8))


def test_seek_is_exact_on_a_real_video(tmp_path) -> None:
    """With real OpenCV decoding, seeks land on the requested frame and stay in sync."""
    path = tmp_path / "numbered.avi"
    _write_numbered_video(path, 120)
    reader = VideoReader(path)

    for target in (0, 1, 59, 60, 61, 100, 116):
        reader.seek(target)
        for expected in range(target, target + 3):
            frame = reader.load_next_frame()
            assert frame["index"] == expected
            assert _decode_number(frame["data"]) == expected

    reader.close()
