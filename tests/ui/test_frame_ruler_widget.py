from unittest import mock

import numpy as np
import pytest

try:
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication

    from jabs.ui.behavior_timeline.behavior_timeline_widget import BehaviorTimelineWidget
    from jabs.ui.behavior_timeline.frame_ruler_widget import FrameRulerWidget
    from jabs.ui.behavior_timeline.track_widgets.timeline_util import (
        frame_pixel_widths,
        window_geometry,
    )

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


@pytest.mark.parametrize("width", [600, 1005, 1200])
def test_ticks_line_up_with_detail_bar_in_timeline_layout(width: int) -> None:
    """In the real timeline layout, each tick sits at the center of its frame in the bars.

    The detail bars are inset by the identity frame and track widget margins, so they
    are narrower than the ruler; the expected positions come from the pixel edges the
    bar itself draws with, mapped into ruler coordinates.
    """
    num_frames = 1000
    current_frame = 500
    pose = mock.Mock(num_identities=2, num_frames=num_frames)
    pose.identity_index_to_display.side_effect = str

    timeline = BehaviorTimelineWidget()
    timeline.pose = pose
    timeline.resize(width, 300)
    timeline.show()
    QApplication.processEvents()

    ruler = timeline._frame_labels
    bar = ruler._reference_bar
    assert bar is not None
    assert bar.width() < ruler.width()

    window_size = ruler._window_size
    frame_width = bar.width() / ruler._nframes
    geometry = window_geometry(current_frame, window_size, num_frames, frame_width)
    edges = geometry.x0 + np.concatenate(
        ([0], np.cumsum(frame_pixel_widths(geometry, frame_width)))
    )

    start = current_frame - window_size
    for frame in range(start, current_frame + window_size + 1, 50):
        k = frame - start
        left = ruler.mapFromGlobal(bar.mapToGlobal(QPoint(int(edges[k]), 0))).x()
        right = ruler.mapFromGlobal(bar.mapToGlobal(QPoint(int(edges[k + 1]), 0))).x()
        assert ruler._frame_center_x(frame, start) == pytest.approx((left + right) / 2)

    timeline.close()


def test_ticks_span_ruler_width_without_reference_bar() -> None:
    """With no reference bar, the window spans the ruler's own width."""
    ruler = FrameRulerWidget()
    ruler.resize(1005, ruler.sizeHint().height())

    # 1005 / 201 = 5 pixels per frame, so window frame 0 spans pixels 0-5
    assert ruler._frame_center_x(400, 400) == pytest.approx(2.5)
    assert ruler._frame_center_x(600, 400) == pytest.approx(1002.5)


def test_ticks_stay_distinct_below_window_frame_count() -> None:
    """A ruler narrower than the 201-frame window still spreads its ticks apart."""
    ruler = FrameRulerWidget()
    ruler.resize(180, ruler.sizeHint().height())

    positions = [ruler._frame_center_x(frame, 400) for frame in (400, 450, 500, 550, 600)]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)


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
