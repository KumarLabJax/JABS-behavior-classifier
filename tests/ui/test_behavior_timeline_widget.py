from unittest import mock

import pytest

try:
    from PySide6.QtWidgets import QApplication

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
