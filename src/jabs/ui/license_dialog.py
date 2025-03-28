from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton)

from ..constants import APP_NAME, APP_NAME_LONG


class LicenseAgreementDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"Accept {APP_NAME_LONG} License")
        self.setModal(True)

        layout = QVBoxLayout()

        layout.addWidget(
            QLabel(f"I have read and I agree to the {APP_NAME} license terms."),
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
