import sys

from PySide6 import QtCore
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
    """Custom progress dialog that allows for cancellation but also doesn't close the progress dialog upon cancellation.

    This dialog emits a signal when the cancel button is clicked, allowing the caller to handle the cancellation and
    close the dialog when appropriate.
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

    def on_cancel(self):
        """Handle the cancel button click."""
        self.canceled.emit()

    def closeEvent(self, event):
        """Ignore close events to prevent user from closing the dialog"""
        event.ignore()

    def setValue(self, value: int) -> None:
        """Set the current value of the progress bar."""
        self._progress.setValue(value)


def create_cancelable_progress_dialog(
    parent: QWidget, text: str, steps: int
) -> CustomProgressDialog:
    """Setup the progress dialog for training or classification tasks.

    This creates a custom cancelable progress dialog. The cancel button behaves different
    from a standard QProgressDialog, as it emits a signal instead of closing the dialog.
    The dialog will be closed by the caller when the task is complete or successfully cancelled.
    """
    dialog = CustomProgressDialog(parent, text, steps)
    dialog.show()
    return dialog


def create_progress_dialog(parent: QWidget, text: str, steps: int) -> QProgressDialog:
    """Create a buttonless progress dialog.

    Parent widget will bet setup as en event filter.
    """
    dialog = QProgressDialog(text, None, 0, steps, parent)
    dialog.installEventFilter(parent)
    dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    _set_window_flags(dialog)

    dialog.show()
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
