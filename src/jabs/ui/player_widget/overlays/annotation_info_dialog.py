from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFormLayout, QLabel, QPushButton, QVBoxLayout


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

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
