import sys

from PySide6 import QtCore
from PySide6.QtWidgets import QProgressDialog, QWidget


def create_progress_dialog(parent: QWidget, text: str, steps: int) -> QProgressDialog:
    """Setup the progress dialog for training or classification tasks."""
    dialog = QProgressDialog(text, None, 0, steps, parent)
    dialog.installEventFilter(parent)
    dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

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

    dialog.show()
    return dialog
