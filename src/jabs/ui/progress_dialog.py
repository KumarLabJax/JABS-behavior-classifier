import sys

from PySide6 import QtCore
from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CustomProgressDialog(QDialog):
    """A custom progress dialog with a cancel button that emits a signal instead of closing.

    This dialog is used for long-running tasks where cancellation is allowed. When the user clicks
    the cancel button, the `canceled` signal is emitted, allowing the caller to handle cancellation
    logic and close the dialog programmatically. The dialog cannot be closed by the user via the
    window close button or the ESC key.

    Signals:
        canceled: Emitted when the cancel button is clicked.

    Args:
        parent (QWidget): The parent widget for the dialog.
        text (str | None): The window title and label text for the dialog.
        steps (int): The maximum value for the progress bar, representing the number of steps.
    """

    canceled = QtCore.Signal()

    def __init__(self, parent: QWidget, text: str | None, steps: int):
        super().__init__(parent)
        self.setWindowTitle(text)
        self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        _set_window_flags(self)

        # widget layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        self._label = QLabel(text, self)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)
        self._progress = QProgressBar(self)
        self._progress.setRange(0, steps)
        layout.addWidget(self._progress)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)  # No extra margin
        button_layout.setSpacing(0)
        button_layout.addStretch()
        self._cancel_button = QPushButton("Cancel", self)
        self._cancel_button.clicked.connect(self.on_cancel)
        button_layout.addWidget(self._cancel_button)
        layout.addLayout(button_layout)

    def on_cancel(self) -> None:
        """Handle the cancel button click."""
        self.canceled.emit()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ignore close events to prevent user from closing the dialog"""
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Ignore ESC key presses to prevent closing the dialog.

        If we want to allow ESC key to close the dialog, we can modify
        this method to call on_cancel() instead of ignoring the event.
        """
        if event.key() == QtCore.Qt.Key.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)

    def setValue(self, value: int) -> None:
        """Set the current value of the progress bar."""
        self._progress.setValue(value)


class _ProgressDialogEventFilter(QObject):
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Filter events for the progress dialog.

        Ignores close events and ESC key presses to prevent the dialog from closing.
        """
        if event.type() == QtCore.QEvent.Type.Close or (
            event.type() == QtCore.QEvent.Type.KeyPress and event.key() == QtCore.Qt.Key.Key_Escape
        ):
            event.ignore()
            return True
        return super().eventFilter(obj, event)


def create_cancelable_progress_dialog(
    parent: QWidget, text: str, steps: int
) -> CustomProgressDialog:
    """Create a cancelable progress dialog for training or classification tasks.

    This creates a custom cancelable progress dialog. The cancel button behaves different
    from a standard QProgressDialog cancel button, as it emits a signal instead of closing the dialog.
    The dialog will be closed by the caller when the task is complete or successfully cancelled.

    The dialog will not be shown immediately, it will be up to the caller to call `show()` on the dialog.
    """
    return CustomProgressDialog(parent, text, steps)


def create_progress_dialog(parent: QWidget, text: str, steps: int) -> QProgressDialog:
    """Create a buttonless progress dialog.

    This can't be cancelled by the user, but it can be closed programmatically.
    The dialog will not be shown immediately, it will be up to the caller to call `show()` on the dialog.
    """
    dialog = QProgressDialog(text, None, 0, steps, parent)
    dialog.installEventFilter(_ProgressDialogEventFilter(dialog))
    dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    _set_window_flags(dialog)
    return dialog


def _set_window_flags(dialog: QDialog) -> None:
    """Set the window flags for the dialog."""
    if sys.platform == "darwin":
        dialog.setWindowFlags(
            QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
    else:
        dialog.setWindowFlags(
            QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.CustomizeWindowHint
        )
