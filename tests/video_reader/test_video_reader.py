"""Tests for VideoReader, in particular that seek() lands on the requested frame."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

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
    ``timestamp_step_ms`` rounds timestamps to that step, like a coarse container time base.

    Each frame's pixels hold its true frame number, so a test can tell which frame it
    actually received.
    """

    def __init__(
        self,
        num_frames: int,
        landing_shift: int = 0,
        reported_count: int | None = None,
        timestamp_step_ms: float | None = None,
    ) -> None:
        self.num_frames = num_frames
        self.landing_shift = landing_shift
        self.reported_count = reported_count or num_frames
        self.timestamp_step_ms = timestamp_step_ms
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
            # like OpenCV, the timestamp of the last decoded frame
            if self._grabbed is None:
                return 0.0
            if self.timestamp_step_ms is None:
                return self._grabbed * _FRAME_MS
            step = self.timestamp_step_ms
            return round(self._grabbed * _FRAME_MS / step) * step
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


def _open_fake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capture: FakeCapture
) -> VideoReader:
    """Open a VideoReader whose OpenCV capture is ``capture``."""
    monkeypatch.setattr(video_reader_module.cv2, "VideoCapture", lambda _path: capture)
    return VideoReader(tmp_path / "video.mp4")


def _assert_plays_in_sync(reader: VideoReader, start: int, frames: int = 3) -> None:
    """Each frame read from ``start`` on is labeled with the frame it really is."""
    for expected in range(start, start + frames):
        frame = reader.load_next_frame()
        assert frame["index"] == expected
        assert _number_in(frame["data"]) == expected, f"labeled {expected}"


def test_seek_corrects_opencv_landing_a_frame_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_seek_corrects_a_short_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Short videos, where the frame rate is off by more, are corrected too."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=20))

    for target in range(17):
        reader.seek(target)
        _assert_plays_in_sync(reader, target)


def test_seek_backs_up_after_overshooting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A landing past the requested frame is retried further back."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100, landing_shift=2))

    for target in (0, 1, 30, 75, 96):
        reader.seek(target)
        _assert_plays_in_sync(reader, target)


def test_seek_past_the_end_reports_end_of_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Seeking beyond the last frame leaves the next read at end of file."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100))

    reader.seek(150)

    assert reader.load_next_frame()["data"] is None


def test_seek_is_exact_when_frame_count_is_an_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Frames are numbered from their timestamps, so a frame count that is off is ignored."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100, reported_count=95))

    for target in (10, 60, 90):
        reader.seek(target)
        _assert_plays_in_sync(reader, target)


def test_seek_trusts_opencv_when_timestamps_are_too_coarse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Without precise enough timing, seek() falls back to OpenCV's position and says so.

    Millisecond timestamps on a 30 fps video can't pin down the frame duration well
    enough to number every frame of a long video.
    """
    capture = FakeCapture(num_frames=100_000, timestamp_step_ms=1.0)
    reader = _open_fake(monkeypatch, tmp_path, capture)

    with caplog.at_level(logging.WARNING, logger=video_reader_module.__name__):
        reader.seek(10)

    assert reader.load_next_frame()["index"] == 10
    assert "Unable to measure frame timing" in caplog.text


def test_seek_corrects_a_video_with_coarse_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Coarse timestamps are still precise enough to number the frames of a short video."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100, timestamp_step_ms=1.0))

    for target in (0, 50, 80, 97):
        reader.seek(target)
        _assert_plays_in_sync(reader, target)


def test_reading_without_seeking_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sequential reads from the start label every frame correctly."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100))

    _assert_plays_in_sync(reader, 0, frames=100)
    assert reader.load_next_frame()["data"] is None


def _write_numbered_video(path: Path, num_frames: int) -> None:
    """Write an MJPEG video whose frames show their frame number as black/white blocks."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), _FPS, (160, 64))
    if not writer.isOpened():
        pytest.skip("MJPG VideoWriter unavailable in this OpenCV build")
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


def test_seek_is_exact_on_a_real_video(tmp_path: Path) -> None:
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


class CountingCapture(FakeCapture):
    """FakeCapture that counts how many frames are decoded."""

    def __init__(self, num_frames: int, reported_count: int | None = None) -> None:
        super().__init__(num_frames, reported_count=reported_count)
        self.grabs = 0

    def grab(self) -> bool:
        """Decode the next frame, counting the call."""
        self.grabs += 1
        return super().grab()


def test_seek_with_frame_count_one_too_high(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A count one higher than the decodable frames doesn't skew the frame numbering.

    Real recordings do this; taking the count at its word numbered every frame past
    the midpoint one too high.
    """
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=1000, reported_count=1001))

    reader.seek(800)
    _assert_plays_in_sync(reader, 800)


def test_seek_to_last_frame_when_landing_overshoots_the_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A landing past the end of the file is retried further back."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100, landing_shift=2))

    reader.seek(99)
    _assert_plays_in_sync(reader, 99, frames=1)


def test_timing_measurement_does_not_decode_rest_of_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An underestimated frame count doesn't make the first seek decode to the end."""
    capture = CountingCapture(num_frames=10_000, reported_count=100)
    reader = _open_fake(monkeypatch, tmp_path, capture)

    with caplog.at_level(logging.WARNING, logger=video_reader_module.__name__):
        reader.seek(10)

    assert capture.grabs < 50
    assert reader.load_next_frame()["index"] == 10
    assert "Unable to find the last frame" in caplog.text


def test_seek_to_a_negative_index_starts_at_the_first_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A negative index is clamped to frame 0 instead of being retried as an overshoot."""
    reader = _open_fake(monkeypatch, tmp_path, FakeCapture(num_frames=100))

    with caplog.at_level(logging.WARNING, logger=video_reader_module.__name__):
        reader.seek(-1)

    _assert_plays_in_sync(reader, 0)
    assert not reader._warned_inexact_seek
    assert caplog.text == ""


def test_seek_does_not_decode_through_a_landing_far_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A landing far before the target falls back to OpenCV instead of decoding forward."""
    capture = CountingCapture(num_frames=10_000)
    reader = _open_fake(monkeypatch, tmp_path, capture)
    reader.seek(0)  # measure the timing first, so only the bad seek is counted
    capture.landing_shift = -1_000
    capture.grabs = 0

    with caplog.at_level(logging.WARNING, logger=video_reader_module.__name__):
        reader.seek(5_000)

    assert capture.grabs <= 2 * video_reader_module._MAX_CATCH_UP_FRAMES
    assert "Unable to seek" in caplog.text
