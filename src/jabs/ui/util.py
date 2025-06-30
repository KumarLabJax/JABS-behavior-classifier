from pathlib import Path

from PySide6.QtCore import QFile


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
    return bool(file.moveToTrash())
