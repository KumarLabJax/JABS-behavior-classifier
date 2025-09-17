import importlib.resources

from PySide6 import QtCore, QtSvg


class EarTagIconManager:
    """Manages SVG icons for ear tags.

    Loads SVG files from the package resources, converts them to a QSvgRenderer
    and provides access to them by ear tag code.
    """

    def __init__(self) -> None:
        self._icons = {}

        base = importlib.resources.files("jabs.resources.eartag_images")
        for entry in base.iterdir():
            if not entry.name.endswith(".svg"):
                continue
            renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(entry.read_bytes()))
            key = entry.name.upper().rsplit(".", 1)[0]
            self._icons[key] = renderer

    def get_icon(self, code: str) -> QtSvg.QSvgRenderer | None:
        """Get the SVG renderer for a given ear tag code.

        Args:
            code: Ear tag code (e.g., "S2")

        Returns:
            QSvgRenderer for the icon, or None if not found
        """
        return self._icons.get(code.upper())
