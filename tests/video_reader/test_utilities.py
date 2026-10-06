"""Tests for the video metadata helpers in :mod:`jabs.video_reader.utilities`."""

from collections.abc import Callable
from pathlib import Path
from unittest import mock

import cv2
import pytest

from jabs.video_reader import VideoReader
from jabs.video_reader.utilities import get_fps_and_nframes


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
def capture(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Patch cv2.VideoCapture in the utilities module and return the fake stream."""
    stream = _fake_capture()
    monkeypatch.setattr(
        "jabs.video_reader.utilities.cv2.VideoCapture", mock.Mock(return_value=stream)
    )
    return stream


def test_get_fps_and_nframes(capture: mock.Mock) -> None:
    """Both properties are read from a single capture, which is then released."""
    assert get_fps_and_nframes("video.avi") == (30, 100)
    capture.isOpened.assert_called_once()
    capture.release.assert_called_once()


def test_video_reader_nframes_from_file(capture: mock.Mock) -> None:
    """VideoReader.get_nframes_from_file returns the frame count from the shared helper."""
    assert VideoReader.get_nframes_from_file(Path("video.avi")) == 100
    capture.release.assert_called_once()


# both entry points take a path; one returns the (fps, nframes) pair, the other just the count
MetadataReader = Callable[[Path], int | tuple[int, int]]


@pytest.mark.parametrize(
    "func",
    [get_fps_and_nframes, VideoReader.get_nframes_from_file],
    ids=["fps_and_nframes", "video_reader"],
)
def test_unopenable_video_raises_and_releases(
    monkeypatch: pytest.MonkeyPatch, func: MetadataReader
) -> None:
    """A video that will not open raises OSError, and the capture is still released."""
    stream = _fake_capture(opened=False)
    monkeypatch.setattr(
        "jabs.video_reader.utilities.cv2.VideoCapture", mock.Mock(return_value=stream)
    )

    with pytest.raises(OSError, match="unable to open"):
        func(Path("missing.avi"))

    stream.release.assert_called_once()
