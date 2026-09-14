"""Tests for the video metadata helpers in :mod:`jabs.video_reader.utilities`."""

from unittest import mock

import cv2
import pytest

from jabs.video_reader import VideoReader
from jabs.video_reader.utilities import get_fps, get_fps_and_nframes, get_frame_count


def _fake_capture(*, opened: bool = True, fps: float = 30.4, nframes: float = 100.0) -> mock.Mock:
    """Build a stand-in for cv2.VideoCapture that reports the given properties."""
    stream = mock.Mock()
    stream.isOpened.return_value = opened
    stream.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: nframes,
    }[prop]
    return stream


@pytest.fixture
def capture(monkeypatch) -> mock.Mock:
    """Patch cv2.VideoCapture in the utilities module and return the fake stream."""
    stream = _fake_capture()
    monkeypatch.setattr(
        "jabs.video_reader.utilities.cv2.VideoCapture", mock.Mock(return_value=stream)
    )
    return stream


def test_get_frame_count(capture: mock.Mock) -> None:
    """The frame count property is returned as an int and the capture is released."""
    assert get_frame_count("video.avi") == 100
    capture.release.assert_called_once()


def test_get_fps_rounds(capture: mock.Mock) -> None:
    """The frame rate is rounded to the nearest int and the capture is released."""
    assert get_fps("video.avi") == 30
    capture.release.assert_called_once()


def test_get_fps_and_nframes(capture: mock.Mock) -> None:
    """Both properties are read from a single capture, which is then released."""
    assert get_fps_and_nframes("video.avi") == (30, 100)
    capture.release.assert_called_once()


def test_video_reader_nframes_matches_helper(capture: mock.Mock) -> None:
    """VideoReader.get_nframes_from_file shares the helper's implementation."""
    assert VideoReader.get_nframes_from_file("video.avi") == 100
    capture.release.assert_called_once()


@pytest.mark.parametrize(
    "func",
    [get_frame_count, get_fps, get_fps_and_nframes, VideoReader.get_nframes_from_file],
    ids=["frame_count", "fps", "fps_and_nframes", "video_reader"],
)
def test_unopenable_video_raises_and_releases(monkeypatch, func) -> None:
    """A video that will not open raises OSError, and the capture is still released."""
    stream = _fake_capture(opened=False)
    monkeypatch.setattr(
        "jabs.video_reader.utilities.cv2.VideoCapture", mock.Mock(return_value=stream)
    )

    with pytest.raises(OSError, match="unable to open"):
        func("missing.avi")

    stream.release.assert_called_once()
