from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qt_material_icons import MaterialIcon

from jabs.project import timeline_annotations

DEFAULT_ANNOTATION_COLOR = "#6495ED"  # cornflower blue

HELP_TEXT = f"""
<b>What are Timeline Annotations?</b><br>
Timeline Annotations are user-defined labels for specific frame intervals in a video. They are not used for model
training; they help flag edge cases, disagreements, or notes for review. Each annotation includes a start and
end frame, a short tag, an optional animal identity, a display color, and an optional free-text description.
<br><br>
<b>Tag requirements</b><br>
  • Alphanumeric only, may include <code>-</code> and <code>_</code>.<br>
  • Length ≤ {timeline_annotations.MAX_TAG_LEN} characters.<br>
  • No whitespace or special characters allowed.<br>
  • Tags are case-preserving for display but are case insensitive for matching and search.<br><br> 
<b>Color</b><br>
  • Use the picker to choose a color. This color is used for the annotation overlay in the video player.<br><br>
<b>Annotation scope</b><br>
  • <i>Selected identity</i>: Annotation applies to the currently selected identity in the main window.<br>
  • <i>Video</i>: Annotation applies to the video and not a specific animal identity.
"""

EDIT_HELP_TEXT = """
<b>Editing an existing annotation</b><br>
  • You may update the <i>tag</i>, <i>color</i>, and <i>description</i> fields.<br>
  • The start and end frames of the interval cannot be modified.<br>
  • The scope (selected identity vs entire video) cannot be modified.<br>
  • To change the interval or scope, please delete this annotation and create a new one.<br><br>
"""


class AnnotationEditDialog(QDialog):
    """Dialog to add a new annotation or edit an existing one.

    Args:
        start: Start frame of the annotation interval.
        end: End frame of the annotation interval.
        parent: Parent widget.
        tag: Initial tag value (for editing existing annotation).
        color: Initial color value (for editing existing annotation).
        description: Initial description value (for editing existing annotation).
        identity_scoped: Scope value (for editing existing annotation).
        identity_index: Index of the identity this annotation applies to (if any).
        display_identity: Display name of the identity (if any).
        edit_mode: If True, the dialog is in edit mode (editing existing annotation).

    Emits:
        annotation_deleted: Emitted when an annotation is deleted, with payload containing
            start, end, tag, and identity_index.

    In edit_mode, you can change tag/color/description or delete. Interval and scope are currently locked
    to the initial values.
    Uniqueness key elsewhere is (start, end, tag, identity_index).

    Note: start/end are UI-inclusive; IntervalTree operations elsewhere account for end+1 (exclusive).
    """

    annotation_deleted = Signal(object)

    def __init__(
        self,
        start: int,
        end: int,
        parent=None,
        *,
        tag: str | None = None,
        color: str | QColor | None = None,
        description: str | None = None,
        identity_scoped: bool | None = None,
        identity_index: int | None = None,
        display_identity: str | None = None,
        edit_mode: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Annotation" if edit_mode else "Add Annotation")
        self._start = start
        self._end = end
        self._identity_index = identity_index
        self._initial_tag_value = tag

        if edit_mode and None in (tag, color, identity_scoped):
            raise ValueError("In edit mode, tag, color, and identity_scoped must be provided.")

        # root layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(16)

        # form area
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(form)

        # Start / End (read-only display on one row each)
        self._start_label = QLabel(str(start))
        self._end_label = QLabel(str(end))
        form.addRow("Start:", self._start_label)
        form.addRow("End:", self._end_label)

        # Optional identity display
        self._identity_name_label = None
        if display_identity:
            self._identity_name_label = QLabel(display_identity)
            self._identity_name_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            form.addRow("Identity:", self._identity_name_label)

        # Tag
        self._tag_edit = QLineEdit()
        self._tag_edit.setMaxLength(timeline_annotations.MAX_TAG_LEN)
        char_width = self._tag_edit.fontMetrics().averageCharWidth()
        self._tag_edit.setMinimumWidth(char_width * (timeline_annotations.MAX_TAG_LEN + 2))
        self._tag_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._tag_edit.textChanged.connect(self._update_tag_label_style)
        form.addRow("Tag:", self._tag_edit)
        if tag:
            self._tag_edit.setText(tag)

        # Color (clickable swatch + hex label)
        self._color = QColor(color) if color else QColor(DEFAULT_ANNOTATION_COLOR)
        color_row = QWidget()
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(8)

        self._color_swatch = QFrame()
        self._color_swatch.setFixedSize(20, 20)
        self._color_swatch.setFrameShape(QFrame.Shape.Box)
        self._color_swatch.setLineWidth(1)
        self._color_swatch.setToolTip("Click to choose a color")
        self._color_swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._color_swatch.installEventFilter(self)

        self._color_label = QLabel()
        self._color_label.setMinimumWidth(80)

        color_layout.addWidget(self._color_swatch)
        color_layout.addWidget(self._color_label)
        color_layout.addStretch(1)

        self._update_color_display()
        form.addRow("Color:", color_row)

        # Description
        self._description_edit = QLineEdit()
        self._description_edit.setPlaceholderText("Optional description…")
        self._description_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        char_width = self._description_edit.fontMetrics().averageCharWidth()
        self._description_edit.setMinimumWidth(char_width * 80)
        if description is not None:
            self._description_edit.setText(description)
        form.addRow("Description:", self._description_edit)

        # annotation scope (identity vs video)
        self._identity_radio = QRadioButton("Selected identity")
        self._video_radio = QRadioButton("Video")
        # Default selection is identity, but allow override via parameter
        if identity_scoped is None:
            self._identity_radio.setChecked(True)
        else:
            self._identity_radio.setChecked(bool(identity_scoped))
            self._video_radio.setChecked(not bool(identity_scoped))
        radio_group = QButtonGroup(self)
        radio_group.addButton(self._identity_radio)
        radio_group.addButton(self._video_radio)

        applies_widget = QWidget()
        applies_vlayout = QVBoxLayout(applies_widget)
        applies_vlayout.setContentsMargins(0, 0, 0, 0)
        applies_vlayout.setSpacing(6)
        applies_vlayout.addWidget(self._identity_radio)
        applies_vlayout.addWidget(self._video_radio)
        form.addRow("Annotation scope:", applies_widget)

        # In edit mode, the scope (identity vs video) cannot be changed
        if edit_mode:
            self._identity_radio.setEnabled(False)
            self._video_radio.setEnabled(False)
            tooltip = "Scope cannot be changed when editing an existing annotation."
            self._identity_radio.setToolTip(tooltip)
            self._video_radio.setToolTip(tooltip)

        # Collapsible Help panel (hidden by default)
        toggle_button = QToolButton()
        toggle_button.setText("Show help")
        toggle_button.setCheckable(True)
        toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        layout.addWidget(toggle_button)

        details_panel = QWidget()
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(8, 8, 8, 8)
        details_layout.setSpacing(6)

        details_label = QLabel()
        details_label.setTextFormat(Qt.TextFormat.RichText)
        details_label.setWordWrap(True)
        details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        details_label.setText(f"""
            {HELP_TEXT}
            {"<br><br>" + EDIT_HELP_TEXT if edit_mode else ""}
        """)
        details_label.setOpenExternalLinks(True)
        details_layout.addWidget(details_label)

        details_panel.setVisible(False)
        layout.addWidget(details_panel)

        def _toggle_help(checked: bool) -> None:
            """Show or hide the details panel and update button text/arrow."""
            details_panel.setVisible(checked)
            toggle_button.setText("Hide help" if checked else "Show help")
            toggle_button.setArrowType(
                Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
            )
            # Recompute layout so the dialog shrinks/expands to content
            details_panel.updateGeometry()
            self.layout().activate()
            self.adjustSize()

        toggle_button.toggled.connect(_toggle_help)

        #  Dialog buttons
        button_row = QHBoxLayout()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        if edit_mode:
            delete_button = QToolButton()
            delete_button.setToolTip("Delete this annotation")
            delete_button.setIcon(MaterialIcon("delete"))
            delete_button.setIconSize(delete_button.iconSize() or QSize(18, 18))
            delete_button.clicked.connect(self._confirm_delete)
            delete_button.setStyleSheet("""
                QToolButton {
                    border-radius: 6px;
                    background-color: transparent;
                    padding: 4px;
                }
                QToolButton:hover {
                    background-color: rgba(0,0,0,0.1);
                }
            """)
            button_row.addWidget(delete_button)

        button_row.addStretch(1)
        button_row.addWidget(buttons)

        layout.addLayout(button_row)

        # Validator wiring
        self._tag_edit.textChanged.connect(self._update_ok_button_state)
        self._update_ok_button_state()

    def get_annotation(self) -> dict:
        """Return the annotation data as a dictionary."""
        return {
            "tag": self._tag_edit.text(),
            "color": self._color.name(QColor.NameFormat.HexRgb),
            "description": self._description_edit.text() or None,
            "identity_scoped": self._identity_radio.isChecked(),
        }

    def _update_color_display(self) -> None:
        """Update the color swatch and label to reflect the current color."""
        name = self._color.name(QColor.NameFormat.HexRgb) if self._color.isValid() else "(none)"
        self._color_label.setText(name)
        self._color_swatch.setStyleSheet(f"background-color: {name};")

    def _pick_color(self) -> None:
        """Open color picker dialog and update color."""
        chosen = QColorDialog.getColor(self._color, self, "Choose Annotation Color")
        if chosen.isValid():
            self._color = chosen
            self._update_color_display()
            self._update_ok_button_state()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        """Handle clicks on the color swatch to open the color picker."""
        if (
            obj is self._color_swatch
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._pick_color()
            return True
        return super().eventFilter(obj, event)

    def _confirm_delete(self) -> None:
        """Confirm deletion; on Yes, emit signal and close dialog.

        Parent code should connect to `annotation_deleted` and remove the interval(s)
        from the appropriate IntervalTree in the VideoLabels object.
        """
        reply = QMessageBox.question(
            self,
            "Delete annotation?",
            "Are you sure you want to delete this annotation? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            payload = {
                "start": self._start,
                "end": self._end,
                "tag": self._initial_tag_value,
                "identity_index": self._identity_index,
            }
            self.annotation_deleted.emit(payload)
            # Close the dialog after emitting
            self.reject()

    @staticmethod
    def _is_tag_valid(tag: str) -> bool:
        """Check if a tag is valid

        Args:
            tag: the tag string to validate
        Returns:
            True if valid, False otherwise
        """
        return 0 < len(tag) <= timeline_annotations.MAX_TAG_LEN and all(
            c.isalnum() or c in "_-" for c in tag
        )

    def _update_tag_label_style(self, tag: str) -> None:
        """Update the tag field style based on validity.

        Args:
            tag: the current tag text
        """
        invalid = not self._is_tag_valid(tag)
        self._tag_edit.setStyleSheet("" if not invalid else "color: red;")
        self._tag_edit.setToolTip(
            ""
            if not invalid
            else f"Alphanumeric, -, _ only; length ≤ {timeline_annotations.MAX_TAG_LEN}"
        )

    def _update_ok_button_state(self) -> None:
        """Enable or disable the OK button based on form validity."""
        tag_valid = self._is_tag_valid(self._tag_edit.text())
        color_valid = self._color.isValid()
        self._ok_button.setEnabled(tag_valid and color_valid)
