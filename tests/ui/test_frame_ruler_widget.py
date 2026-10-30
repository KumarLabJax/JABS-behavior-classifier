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


@pytest.mark.parametrize("width", [600, 1000, 1005, 1200, 180])
def test_frame_center_matches_detail_bar_mapping(width: int) -> None:
    """Ticks use the detail bar's fractional pixels-per-frame, with no centering offset.

    LabelDetailBar draws window frame ``k`` from ``k * width / 201`` pixels, so a tick
    must sit at the center of that span at every width, not only at widths that are a
    multiple of 201 pixels.
    """
    ruler = FrameRulerWidget()
    ruler.resize(width, ruler.sizeHint().height())
    frame_width = width / 201

    # window of frames 400-600 centered on frame 500
    for frame in (400, 450, 500, 550, 600):
        assert ruler._frame_center_x(frame, 400) == pytest.approx(
            (frame - 400 + 0.5) * frame_width
        )


def test_ticks_stay_distinct_below_window_frame_count() -> None:
    """A ruler narrower than the 201-frame window still spreads its ticks apart."""
    ruler = FrameRulerWidget()
    ruler.resize(180, ruler.sizeHint().height())

    positions = [ruler._frame_center_x(frame, 400) for frame in (400, 450, 500, 550, 600)]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)
