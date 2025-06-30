from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)


class ProjectPruningConfirmationDialog(QDialog):
    """Dialog that displays a list of videos to be pruned from the project and asks the user for confirmation.

    Presents the user with a scrollable list of video file names and
    OK/Cancel buttons to confirm or abort the pruning operation.
    """

    def __init__(self, videos: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Videos to Prune")

        layout = QVBoxLayout(self)

        info_text = "Click OK to remove the following videos from the project:"
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.list_widget = QListWidget()
        self.list_widget.addItems(videos)
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.ok_button = QPushButton("OK")
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        layout.addLayout(button_layout)

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
