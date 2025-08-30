import traceback
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)
from qt_material_icons import MaterialIcon

from jabs.ui.annotation_edit_dialog import AnnotationEditDialog
from jabs.ui.util import find_central_widget

if TYPE_CHECKING:
    from jabs.ui.central_widget import CentralWidget


class AnnotationInfoDialog(QDialog):
    """Dialog to display detailed information about an annotation."""

    def __init__(self, annotation_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annotation Details")
        layout = QVBoxLayout(self)

        tag_label = QLabel(f"<b>{annotation_data.get('tag', '')}</b>")
        tag_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(tag_label)

        form = QFormLayout()
        form.addRow("Start Frame:", QLabel(str(annotation_data.get("start", ""))))
        form.addRow("End Frame:", QLabel(str(annotation_data.get("end", ""))))
        desc_label = QLabel(annotation_data.get("description", ""))
        desc_label.setWordWrap(True)
        form.addRow("Description:", desc_label)
        layout.addLayout(form)

        # Action buttons row: Edit (pencil) and Close
        button_row = QHBoxLayout()
        edit_button = QToolButton()
        edit_button.setIcon(MaterialIcon("edit"))
        edit_button.setToolTip("Edit this annotation")
        edit_button.setAutoRaise(True)
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
            raise RuntimeWarning("Error: Could not find central widget to handle annotation edit.")

        # capture the original key so central widget can find/remove the old interval
        original_key = {
            "start": int(data.get("start", 0)),
            "end": int(data.get("end", 0)),
            "tag": data.get("tag"),
            "identity": data.get("identity"),  # None means applies to whole video
        }

        def _on_deleted(payload: dict) -> None:
            central_widget.on_annotation_deleted(payload)

        def launch_editor():
            dialog = AnnotationEditDialog(
                start=original_key["start"],
                end=original_key["end"],
                tag=original_key["tag"],
                color=data.get("color"),
                description=data.get("description"),
                applies_to_identity=bool(data.get("applies_to_identity", True)),
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
                traceback.print_exc()

        # Close this dialog's modal session cleanly, then schedule the editor
        self.accept()
        QTimer.singleShot(0, launch_editor)
