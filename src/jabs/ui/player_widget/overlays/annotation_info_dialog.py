from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qt_material_icons import MaterialIcon

from jabs.ui.annotation_edit_dialog import AnnotationEditDialog
from jabs.ui.util import find_central_widget

if TYPE_CHECKING:
    from jabs.ui.central_widget import CentralWidget

# Swatch size constant for color display
SWATCH_SIZE = 20


class AnnotationInfoDialog(QDialog):
    """Dialog to display detailed information about an annotation.

    Args:
        annotation_data (dict): Dictionary containing annotation details.
        parent (QWidget | None): Parent widget for the dialog.
    """

    def __init__(self, annotation_data: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annotation Details")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)

        tag_label = QLabel(f"<b>{annotation_data.get('tag', '')}</b>")
        tag_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        tag_label.setStyleSheet("font-size: 16pt;")
        layout.addWidget(tag_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.addRow("Start Frame:", QLabel(str(annotation_data.get("start", ""))))
        form.addRow("End Frame:", QLabel(str(annotation_data.get("end", ""))))

        if annotation_data.get("display_identity") is not None:
            form.addRow("Identity:", QLabel(str(annotation_data["display_identity"])))

        # color: show swatch with the annotation color
        color = annotation_data["color"]
        color_row_widget = QWidget()
        color_row_layout = QHBoxLayout(color_row_widget)
        color_row_layout.setContentsMargins(0, 0, 0, 0)
        color_swatch = QLabel()
        pixmap = QPixmap(SWATCH_SIZE, SWATCH_SIZE)
        pixmap.fill(QColor(color))
        color_swatch.setPixmap(pixmap)
        color_row_layout.addWidget(color_swatch)
        color_row_layout.addStretch(1)
        form.addRow("Color:", color_row_widget)

        # description
        description_label = QLabel(annotation_data.get("description", ""))
        description_label.setWordWrap(True)
        form.addRow("Description:", description_label)
        layout.addLayout(form)

        # buttons
        button_row = QHBoxLayout()
        button_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        edit_button = QToolButton()
        edit_button.setIcon(MaterialIcon("edit"))
        edit_button.setIconSize(QSize(24, 24))
        edit_button.setToolTip("Edit this annotation")
        edit_button.setStyleSheet("""
                QToolButton {
                    border-radius: 6px;
                    background-color: transparent;
                    padding: 4px;
                }
                QToolButton:hover {
                    background-color: rgba(0,0,0,0.1);
                }
            """)
        edit_button.clicked.connect(lambda: self._open_editor(annotation_data))

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setDefault(True)

        button_row.addWidget(edit_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _open_editor(self, data: dict) -> None:
        """Open the AnnotationDialog in editable mode with prefilled fields.

        End this dialog's modal session first, then open the editor to avoid
        re-entrant modal warnings on macOS.
        """
        parent = self.parent()
        central_widget: CentralWidget | None = find_central_widget(self)

        if central_widget is None:
            raise RuntimeError("Error: Could not find central widget to handle annotation edit.")

        # capture the original key so central widget can find/remove the old interval
        original_key = {
            "start": int(data.get("start", 0)),
            "end": int(data.get("end", 0)),
            "tag": data.get("tag"),
            "identity": data.get("identity"),  # None means applies to whole video
        }
        identity_scoped = data.get("identity") is not None

        def _on_deleted(payload: dict) -> None:
            central_widget.on_annotation_deleted(payload)

        def launch_editor():
            dialog = AnnotationEditDialog(
                start=original_key["start"],
                end=original_key["end"],
                tag=original_key["tag"],
                color=data.get("color"),
                description=data.get("description"),
                identity_scoped=identity_scoped,
                identity_index=data.get("identity"),  # your existing param
                display_identity=data.get("display_identity"),
                edit_mode=True,
                parent=parent,
            )
            dialog.annotation_deleted.connect(_on_deleted)
            try:
                rc = dialog.exec()
                if rc == QDialog.DialogCode.Accepted:
                    updated = dialog.get_annotation()
                    if central_widget:
                        central_widget.on_annotation_edited(original_key, updated)
            except Exception as e:
                print(f"Error during edit flow: {e}")
                import traceback

                traceback.print_exc()

        # Close this dialog's modal session cleanly, then schedule the editor
        self.accept()
        QTimer.singleShot(0, launch_editor)
