"""Tests for the thread that decodes frames for the player."""

import numpy as np
import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.player_widget.player_thread import PlayerThread

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists; QThread and its signals need one."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class FakeReader:
    """Video reader stand-in handing out numbered frames."""

    def __init__(self, start: int = 0) -> None:
        self.index = start

    def load_next_frame(self) -> dict:
        """Return the next frame, in the shape the reader produces."""
        frame = {
            "data": np.full((8, 8, 3), 60, dtype=np.uint8),
            "index": self.index,
            "duration": 1 / 30,
        }
        self.index += 1
        return frame

    def seek(self, position: int) -> None:
        """Move to a frame, as the real reader does."""
        self.index = position


def test_the_frame_index_travels_with_the_image() -> None:
    """The overlays are drawn from the number the widget is handed with the image.

    Reading the number from a separate signal instead would draw them against
    whichever frame arrived last, which during playback is the previous one.
    """
    thread = PlayerThread(FakeReader(start=7), pose_est=None, identity=0)
    received: list[tuple[object, int]] = []
    thread.newImage.connect(lambda image, index: received.append((image, index)))

    thread._read_and_emit_frame()

    assert len(received) == 1
    image, index = received[0]
    assert index == 7
    assert not image.isNull()


def test_seek_emits_the_frame_it_sought_to() -> None:
    """Stepping a frame reports the frame it landed on, image and position alike."""
    thread = PlayerThread(FakeReader(), pose_est=None, identity=0)
    images: list[int] = []
    positions: list[int] = []
    thread.newImage.connect(lambda _image, index: images.append(index))
    thread.updatePosition.connect(positions.append)

    thread.seek(42)

    assert images == [42]
    assert positions == [42]
