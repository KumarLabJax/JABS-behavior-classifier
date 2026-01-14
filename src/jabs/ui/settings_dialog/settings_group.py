from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from .collapsible_section import CollapsibleSection


class SettingsGroup(QGroupBox):
    """
    A reusable settings group with a grid layout for controls and optional collapsible documentation.

    This class provides a structured layout for adding related settings controls in a grid format,
    with an optional collapsible help/documentation section below the controls.

    The grid layout has three columns:
    - Column 0: Labels (natural size, right-aligned)
    - Column 1: Input widgets (natural size)
    - Column 2: Spacer (expands to fill remaining width, keeping controls left-aligned)

    Subclasses should override `_create_controls()` to add their specific settings widgets
    and optionally override `_create_documentation()` to provide help text.

    Example:
        class MySettingsGroup(SettingsGroup):
            def __init__(self, parent=None):
                super().__init__("My Settings", parent)

            def _create_controls(self):
                self._my_checkbox = QCheckBox()
                self.add_control_row("Enable feature:", self._my_checkbox)

            def _create_documentation(self):
                help_label = QLabel("This is help text...")
                help_label.setWordWrap(True)
                return help_label

            def get_values(self):
                return {"my_setting": self._my_checkbox.isChecked()}

            def set_values(self, values):
                self._my_checkbox.setChecked(values.get("my_setting", False))
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """
        Initialize the settings group.

        Args:
            title: The title displayed in the group box header.
            parent: Parent widget for this settings group.
        """
        super().__init__(title, parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Main vertical layout for the group
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(12, 12, 12, 12)
        self._main_layout.setSpacing(8)

        # Grid layout for form controls
        self._form_widget = QWidget(self)
        self._grid_layout = QGridLayout(self._form_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(12)
        self._grid_layout.setVerticalSpacing(8)
        self._grid_layout.setColumnStretch(0, 0)  # labels column: natural size
        self._grid_layout.setColumnStretch(1, 0)  # inputs column: natural size
        self._grid_layout.setColumnStretch(2, 1)  # consume extra width on the right

        self._main_layout.addWidget(self._form_widget)

        # Track current row for adding controls
        self._current_row = 0

        # Create controls (subclasses override this)
        self._create_controls()

        # Add horizontal spacer in column 2 for all rows
        if self._current_row > 0:
            self._grid_layout.addItem(
                QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum),
                0,
                2,
                self._current_row,
                1,
            )

        # Create optional documentation section
        doc_widget = self._create_documentation()
        if doc_widget is not None:
            self._help_section = CollapsibleSection("What do these do?", doc_widget, self)
            self._help_section.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
            )
            doc_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

            # Connect size change signal for reflow
            self._help_section.sizeChanged.connect(self._on_help_section_resized)

            # Connect toggle signal to scroll to section when expanded
            self._help_section.toggled.connect(self._on_help_toggled)

            self._main_layout.addWidget(self._help_section)
        else:
            self._help_section = None

        self._main_layout.addStretch(0)

    def add_control_row(
        self,
        label_text: str,
        widget: QWidget,
        alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
    ) -> int:
        """
        Add a control row to the grid layout.

        Args:
            label_text: Text for the label in column 0.
            widget: The input widget to add in column 1.
            alignment: Alignment for the widget in column 1 (default: AlignLeft).

        Returns:
            The row index where the control was added.
        """
        label = QLabel(label_text, self)
        self._grid_layout.addWidget(label, self._current_row, 0, Qt.AlignmentFlag.AlignRight)
        self._grid_layout.addWidget(widget, self._current_row, 1, alignment)
        row = self._current_row
        self._current_row += 1
        return row

    def add_widget_row(self, widget: QWidget, column_span: int = 3) -> int:
        """
        Add a widget that spans multiple columns.

        Useful for widgets that don't fit the label/control pattern.

        Args:
            widget: The widget to add.
            column_span: Number of columns to span (default: 3 for full width).

        Returns:
            The row index where the widget was added.
        """
        self._grid_layout.addWidget(widget, self._current_row, 0, 1, column_span)
        row = self._current_row
        self._current_row += 1
        return row

    def _create_controls(self) -> None:
        """
        Create and add control widgets to the settings group.

        Subclasses should override this method to add their specific controls using
        `add_control_row()` or `add_widget_row()`.
        """
        pass

    def _create_documentation(self) -> QWidget | None:
        """
        Create documentation/help content for this settings group.

        Subclasses should override this method to return a widget containing help text
        or documentation. If None is returned, no collapsible documentation section is shown.

        Returns:
            A widget containing documentation, or None if no documentation is needed.
        """
        return None

    def _on_help_section_resized(self) -> None:
        """Handle help section size changes to trigger parent layout updates.

        With the scroll area's setWidgetResizable(True), the layout system
        automatically handles size changes without needing explicit adjustSize calls.
        We just need to activate the layout to reflow content.
        """
        # Just activate layouts without calling adjustSize to avoid shrinking
        parent = self.parentWidget()
        if parent is not None:
            parent_layout = parent.layout()
            if parent_layout is not None:
                parent_layout.activate()

    def _on_help_toggled(self, checked: bool) -> None:
        """
        Handle help section toggle.

        Args:
            checked: True if the help section is expanded, False if collapsed.
        """
        if checked:
            self.scroll_to_help_section()

    def get_scroll_area(self):
        """
        Find the parent QScrollArea if one exists.

        Returns:
            The parent QScrollArea, or None if not found.
        """
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    def scroll_to_help_section(self) -> None:
        """Scroll to make the help section visible in the parent scroll area."""
        if self._help_section is None:
            return

        scroll = self.get_scroll_area()
        if scroll is not None:
            # Defer one tick so QScrollArea can recompute its scroll range correctly
            QTimer.singleShot(0, lambda: scroll.ensureWidgetVisible(self._help_section))

    def get_values(self) -> dict:
        """
        Get the current values from this settings group.

        Subclasses should override this to return a dictionary of setting names to values.

        Returns:
            Dictionary mapping setting names to their current values.
        """
        return {}

    def set_values(self, values: dict) -> None:
        """
        Set values in this settings group.

        Subclasses should override this to update their controls from the provided values.

        Args:
            values: Dictionary mapping setting names to values.
        """
        pass
