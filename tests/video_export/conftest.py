"""Fixtures shared by the video-export tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ._fakes import BACKGROUND, FRAMES, HEIGHT, WIDTH


@pytest.fixture
def blank_frame() -> np.ndarray:
    """A uniform BGR frame, so any change is attributable to the overlay."""
    return np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)


@pytest.fixture
def source_video(tmp_path: Path) -> Path:
    """A short uniform video on disk for the exporter to read."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (WIDTH, HEIGHT))
    for _ in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8))
    writer.release()
    return path
