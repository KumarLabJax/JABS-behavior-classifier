from PySide6.QtCore import Signal
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A collapsible section with a header ToolButton and a content area.

    This widget is used by SettingsGroup to provide inline documentation that is collapsed (hidden) by default.
    The parent dialog's scroll area will handle scrolling when expanded.
    """

    sizeChanged = Signal()
    toggled = Signal(bool)  # Emitted when the section is expanded/collapsed

    def __init__(self, title: str, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content = content
        self._toggle_btn = QToolButton(self)
        self._toggle_btn.setStyleSheet("QToolButton { border: none; }")
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_btn.setText(title)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.toggled.connect(self._on_toggled)

        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)

        # Set size policies to allow vertical expansion
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._content.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self._content.setVisible(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._toggle_btn)
        lay.addWidget(line)
        lay.addWidget(self._content)

    def _on_toggled(self, checked: bool) -> None:
        """Handle toggling the collapsible section."""
        self._toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)
        self._content.updateGeometry()

        # Ask ancestors to recompute layout so the page grows inside the scroll area
        parent = self.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().activate()

        if self.layout() is not None:
            self.layout().activate()

        # Let ancestors recompute size hints and notify listeners
        if parent is not None:
            parent.updateGeometry()
        self.updateGeometry()
        self.sizeChanged.emit()
        self.toggled.emit(checked)

    def is_expanded(self) -> bool:
        """
        Check if the section is currently expanded.

        Returns:
            True if the section is expanded, False otherwise.
        """
        return self._toggle_btn.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        """
        Set the expanded state of the section.

        Args:
            expanded: True to expand the section, False to collapse it.
        """
        self._toggle_btn.setChecked(expanded)
