import importlib.resources

from PySide6.QtSvg import QSvgRenderer


class EarTagIconManager:
    """Manages SVG icons for ear tags.

    Loads SVG files from the package resources, converts them to a QSvgRenderer
    and provides access to them by ear tag code.
    """

    def __init__(self) -> None:
        self._icons: dict[str, QSvgRenderer] = {}

        base = importlib.resources.files("jabs.resources.eartag_images")
        for entry in base.iterdir():
            if not entry.name.endswith(".svg"):
                continue
            renderer = QSvgRenderer(entry.read_bytes())
            key = entry.stem.upper()  # type: ignore
            self._icons[key] = renderer

    def get_icon(self, code: str) -> QSvgRenderer | None:
        """Get the SVG renderer for a given ear tag code.

        Args:
            code: Ear tag code (e.g., "S2")

        Returns:
            QSvgRenderer for the icon, or None if not found
        """
        return self._icons.get(code.upper())
