from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton)

from src import APP_NAME, APP_NAME_LONG


class DisclaimerDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"{APP_NAME_LONG} License Disclaimer")
        self.setModal(True)

        layout = QVBoxLayout()

        layout.addWidget(
            QLabel(f"I have read and agree to the {APP_NAME} license terms."),
            alignment=Qt.AlignCenter
        )

        button_layout = QHBoxLayout()

        yes_button = QPushButton("YES")
        yes_button.clicked.connect(self.accept)

        no_button = QPushButton("NO")
        no_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(yes_button, alignment=Qt.AlignRight)
        button_layout.addWidget(no_button, alignment=Qt.AlignRight)

        layout.addLayout(button_layout)

        self.setLayout(layout)
