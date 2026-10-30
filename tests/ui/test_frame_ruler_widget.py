import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.behavior_timeline.frame_ruler_widget import FrameRulerWidget

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
    """Ensure a QApplication exists for UI-related tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.mark.parametrize(
    ("frame", "expected"),
    [(-50, False), (0, True), (25, False), (950, True), (999, False), (1000, False)],
    ids=["before_start", "first_frame", "between_ticks", "last_tick", "last_frame", "past_end"],
)
def test_has_tick_only_on_existing_frames(frame: int, expected: bool) -> None:
    """Ticks fall every 50 frames and never past the last frame of the video."""
    ruler = FrameRulerWidget()
    ruler.set_num_frames(1000)

    assert ruler._has_tick(frame) is expected
