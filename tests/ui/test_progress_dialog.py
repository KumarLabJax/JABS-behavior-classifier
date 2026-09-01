"""Tests for the custom progress dialog's dismissal contract.

These pin behavior that is deliberate but surprising: the dialog cannot be closed by
the user, which also means ``close()`` does nothing for the code that owns it. Getting
this wrong strands the dialog on screen with no way to dismiss it, so it is worth a
test rather than a comment.
"""

import pytest

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication, QWidget

    from jabs.ui.dialogs.progress_dialog import (
        CustomProgressDialog,
        create_cancelable_progress_dialog,
    )

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def dialog(qapp):
    """A shown cancelable progress dialog with a live parent widget."""
    parent = QWidget()
    parent.show()
    d = create_cancelable_progress_dialog(parent, "Working", 10)
    d.show()
    qapp.processEvents()
    yield d
    d.hide()
    d.deleteLater()
    parent.close()


def test_close_does_not_dismiss_the_dialog(dialog, qapp) -> None:
    """close() is intentionally ignored, so it cannot be used to dismiss the dialog.

    This is the trap: a caller that only calls close() leaves the dialog on screen
    forever, with its cancel button wired to an already-finished task.
    """
    dialog.close()
    qapp.processEvents()

    assert dialog.isVisible()


def test_hide_dismisses_the_dialog(dialog, qapp) -> None:
    """hide() is the way to actually take the dialog down."""
    dialog.hide()
    qapp.processEvents()

    assert not dialog.isVisible()


def test_escape_does_not_dismiss_the_dialog(dialog, qapp) -> None:
    """ESC is ignored, so a stray keypress cannot abandon a running task."""
    dialog.keyPressEvent(
        QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    )
    qapp.processEvents()

    assert dialog.isVisible()


def test_cancel_button_emits_without_dismissing(dialog, qapp) -> None:
    """Cancel signals the owner rather than closing, so teardown stays ordered."""
    emitted: list[bool] = []
    dialog.canceled.connect(lambda: emitted.append(True))

    dialog.on_cancel()
    qapp.processEvents()

    assert emitted == [True]
    assert dialog.isVisible(), "cancel should leave the dialog up until the owner removes it"


def test_set_value_tracks_progress(dialog) -> None:
    """setValue drives the embedded progress bar rather than closing at maximum."""
    dialog.setValue(10)

    assert isinstance(dialog, CustomProgressDialog)
    assert dialog.maximum() == 10
    assert dialog.isVisible(), "unlike QProgressDialog, this one does not auto-close"
