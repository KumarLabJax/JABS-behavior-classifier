import pytest

try:
    from PySide6.QtWidgets import QApplication

    from jabs.ui.behavior_timeline.track_widgets.label_overview_bar import LabelOverviewBar

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
    ("num_frames", "width", "current_frame"),
    [(10000, 700, 5000), (1000, 700, 500), (54000, 1213, 20017), (300, 1000, 150)],
)
def test_highlight_span_covers_the_detail_window(
    num_frames: int, width: int, current_frame: int
) -> None:
    """Both edges of the highlight are mapped from the window's first and last frames."""
    bar = LabelOverviewBar()
    bar._current_frame = current_frame
    frames_per_pixel = num_frames / width
    window_size = bar._window_size

    start, highlight_width = bar._highlight_span(frames_per_pixel)

    assert start == int((current_frame - window_size) / frames_per_pixel)
    assert start + highlight_width == int((current_frame + window_size + 1) / frames_per_pixel)


def test_highlight_span_not_shifted_by_separate_rounding() -> None:
    """Rounding the half-width separately no longer shifts the highlight a pixel.

    With 3000 frames over 1000 pixels and the current frame at 114, the window covers
    frames 14-214, which map to pixels 4-71. Subtracting the separately rounded
    half-width (33) from the current frame's pixel (38) put the box at 5-72 instead.
    """
    bar = LabelOverviewBar()
    bar._current_frame = 114

    assert bar._highlight_span(3000 / 1000) == (4, 67)
