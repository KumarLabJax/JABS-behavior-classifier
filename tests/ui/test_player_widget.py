from pathlib import Path

import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.player_widget import PlayerWidget

    SKIP_UI_TESTS = False
    SKIP_REASON = None
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_reset_clears_current_video_path():
    """Reset restores the documented no-video-loaded state."""
    widget = PlayerWidget()
    widget._video_path = Path("example.mp4")

    widget.reset()

    assert widget.current_video_path is None
