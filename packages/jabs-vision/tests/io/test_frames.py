"""Tests for the imageio-based frame reader."""

from unittest import mock

import numpy as np

import jabs.vision.io.frames as frames_mod
from jabs.vision.io import read_frames, video_fps


class _FakeReader:
    """Minimal stand-in for an imageio ffmpeg reader."""

    def __init__(self, frames, meta):
        self._frames = frames
        self._meta = meta
        self.closed = False

    def iter_data(self):
        """Yield the fixture frames."""
        yield from self._frames

    def get_meta_data(self):
        """Return the fixture metadata."""
        return self._meta

    def close(self):
        """Record that the reader was closed."""
        self.closed = True


def test_read_frames_yields_all_frames(monkeypatch):
    """read_frames yields every frame and closes the reader."""
    fake_frames = [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)]
    reader = _FakeReader(fake_frames, {"fps": 30.0})
    monkeypatch.setattr(frames_mod.imageio, "get_reader", mock.Mock(return_value=reader))

    out = list(read_frames("video.mp4"))

    assert len(out) == 2
    assert out[0].shape == (4, 4, 3)
    assert reader.closed is True


def test_video_fps_rounds(monkeypatch):
    """video_fps rounds the reported frame rate to an int."""
    reader = _FakeReader([], {"fps": 29.97})
    monkeypatch.setattr(frames_mod.imageio, "get_reader", mock.Mock(return_value=reader))
    assert video_fps("video.mp4") == 30


def test_video_fps_defaults_to_30(monkeypatch):
    """video_fps defaults to 30 when the metadata has no fps."""
    reader = _FakeReader([], {})  # no fps key
    monkeypatch.setattr(frames_mod.imageio, "get_reader", mock.Mock(return_value=reader))
    assert video_fps("video.mp4") == 30
