from pathlib import Path

from PySide6.QtCore import QFile
from PySide6.QtWidgets import QMainWindow, QWidget


def send_file_to_recycle_bin(file_path: Path) -> bool:
    """Send a file to the recycle bin.

    Args:
        file_path (Path): The path to the file to be sent to the recycle bin.

    Returns:
        bool: True if the file was successfully sent to the recycle bin, False otherwise.

    Ignores missing files and returns True in that case.
    """
    if not file_path.exists():
        return True  # Ignore missing files
    file = QFile(str(file_path))
    return file.moveToTrash()


def find_central_widget(widget: QWidget) -> QWidget | None:
    """Traverse up the parent hierarchy to find the central widget of the main window.

    Args:
        widget (QWidget): The starting widget to begin the search from.

    Returns:
        QWidget | None: The central widget if found, otherwise None.
    """
    w = widget
    while w is not None:
        if isinstance(w, QMainWindow):
            return w.centralWidget()
        w = w.parent()
    return None
