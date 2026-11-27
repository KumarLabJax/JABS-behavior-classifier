from unittest import mock

import pytest

try:
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication, QPushButton

    from jabs.ui.behavior_timeline.behavior_timeline_widget import BehaviorTimelineWidget

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


def test_binary_prediction_bar_compact_follows_identity_mode() -> None:
    """Switching identity mode after load resizes the binary prediction bar.

    The widgets are created while in All Animals mode (compact); switching to
    Selected Animal must expand them without needing a video reload, and switching
    back must compact them again.
    """
    pose = mock.Mock(num_identities=2, num_frames=100)
    pose.identity_index_to_display.side_effect = str

    timeline = BehaviorTimelineWidget()
    timeline.identity_mode = BehaviorTimelineWidget.IdentityMode.ALL
    timeline.pose = pose
    timeline.active_identity_index = 0

    pred_widget = timeline._prediction_overview_widgets[0][0]
    assert timeline._label_overview_widgets[0].compact
    assert pred_widget.compact

    timeline.identity_mode = BehaviorTimelineWidget.IdentityMode.ACTIVE
    assert not timeline._label_overview_widgets[0].compact
    assert not pred_widget.compact

    timeline.identity_mode = BehaviorTimelineWidget.IdentityMode.ALL
    assert timeline._label_overview_widgets[0].compact
    assert pred_widget.compact

    timeline.close()


def _send_mouse_press(widget, button: Qt.MouseButton) -> None:
    """Dispatch a synthetic MouseButtonPress at the center of ``widget``."""
    pos = QPointF(max(widget.width() / 2.0, 1.0), max(widget.height() / 2.0, 1.0))
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        pos,
        pos,
        pos,
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)


def test_clicking_ethogram_track_emits_identity_selected_in_all_mode() -> None:
    """Left-clicking an animal's label or prediction track in ALL mode selects that animal."""
    pose = mock.Mock(num_identities=3, num_frames=100)
    pose.identity_index_to_display.side_effect = str

    timeline = BehaviorTimelineWidget()
    timeline.identity_mode = BehaviorTimelineWidget.IdentityMode.ALL
    timeline.pose = pose
    timeline.show()

    emitted: list[int] = []
    timeline.identity_selected.connect(emitted.append)

    _send_mouse_press(timeline._label_overview_widgets[1], Qt.MouseButton.LeftButton)
    _send_mouse_press(timeline._prediction_overview_widgets[2][0], Qt.MouseButton.LeftButton)
    # Right-click should not trigger identity selection
    _send_mouse_press(timeline._label_overview_widgets[0], Qt.MouseButton.RightButton)
    assert emitted == [1, 2]

    # Switching to ACTIVE mode ignores track-widget click filtering
    emitted.clear()
    timeline.identity_mode = BehaviorTimelineWidget.IdentityMode.ACTIVE
    _send_mouse_press(timeline._label_overview_widgets[0], Qt.MouseButton.LeftButton)
    assert emitted == []

    timeline.close()


def test_clicking_identity_header_and_selector_strip_syncs_active_highlight() -> None:
    """Header buttons and the bottom Animal selector bar emit once and track active state."""
    pose = mock.Mock(num_identities=3, num_frames=100)
    pose.identity_index_to_display.side_effect = str

    timeline = BehaviorTimelineWidget()
    timeline.identity_mode = BehaviorTimelineWidget.IdentityMode.ALL
    timeline.pose = pose

    emitted: list[int] = []
    timeline.identity_selected.connect(emitted.append)

    # Clicking row header emits once without double-firing from eventFilter
    header_buttons = timeline._identity_frames[2].findChildren(QPushButton)
    assert len(header_buttons) == 1
    assert header_buttons[0].text() == "Animal 2 [2]:"
    header_buttons[0].click()
    assert emitted == [2]

    # Bottom selector bar works in both ALL and ACTIVE modes and tracks active_identity_index
    selector = timeline._identity_selector_widget
    assert selector is not None
    assert selector._buttons[0].text().startswith("* ")
    timeline.active_identity_index = 1
    assert selector._buttons[1].text().startswith("* ")
    assert not selector._buttons[0].text().startswith("* ")

    timeline.identity_mode = BehaviorTimelineWidget.IdentityMode.ACTIVE
    selector._buttons[2].click()
    assert emitted == [2, 2]

    # Resetting pose clears stale click targets and rebuilds selector buttons cleanly
    pose2 = mock.Mock(num_identities=2, num_frames=100)
    pose2.identity_index_to_display.side_effect = str
    timeline.pose = pose2
    assert timeline._identity_selector_widget is not None
    assert set(timeline._identity_selector_widget._buttons.keys()) == {0, 1}

    timeline.close()
