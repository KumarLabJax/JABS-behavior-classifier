"""Tests for the dialog that chooses which overlays go into an exported video."""

import pytest

try:
    from PySide6.QtWidgets import QApplication, QDialogButtonBox

    from jabs.ui.dialogs import VideoExportOptionsDialog

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for UI-related tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _save_button(dialog: "VideoExportOptionsDialog"):
    return dialog._button_box.button(QDialogButtonBox.StandardButton.Save)


def test_initial_states_come_from_the_caller() -> None:
    """The remembered choices from the last export are restored."""
    dialog = VideoExportOptionsDialog(
        draw_pose=False, draw_segmentation=True, draw_predictions=True
    )

    assert dialog.draw_pose is False
    assert dialog.draw_segmentation is True
    assert dialog.draw_predictions is True


def test_unavailable_overlay_is_disabled_with_a_reason() -> None:
    """An overlay the video cannot provide is explained rather than silently missing."""
    dialog = VideoExportOptionsDialog(
        draw_segmentation=True,
        draw_predictions=True,
        segmentation_unavailable="No segmentation data available: needs pose v6",
        predictions_unavailable="No predictions for this video: classify it first",
    )

    assert dialog.draw_segmentation is False
    assert dialog.segmentation_enabled is False
    assert "pose v6" in dialog._segmentation_checkbox.toolTip()
    assert dialog.draw_predictions is False
    assert dialog.predictions_enabled is False
    assert "classify" in dialog._predictions_checkbox.toolTip()


def test_save_is_blocked_until_an_overlay_is_selected() -> None:
    """An export with no overlays is just a slow re-encode, so Save waits for one."""
    dialog = VideoExportOptionsDialog(
        draw_pose=False, draw_segmentation=False, draw_predictions=False
    )

    assert _save_button(dialog).isEnabled() is False

    dialog._predictions_checkbox.setChecked(True)

    assert _save_button(dialog).isEnabled() is True


def test_save_is_blocked_again_when_the_last_overlay_is_cleared() -> None:
    """Unticking the last box disables Save rather than exporting nothing."""
    dialog = VideoExportOptionsDialog(
        draw_pose=True, draw_segmentation=False, draw_predictions=False
    )
    assert _save_button(dialog).isEnabled() is True

    dialog._pose_checkbox.setChecked(False)

    assert _save_button(dialog).isEnabled() is False
