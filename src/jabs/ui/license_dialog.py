from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ..constants import APP_NAME, APP_NAME_LONG


class LicenseAgreementDialog(QDialog):
    """Dialog for accepting the application license agreement.

    Presents the user with a message to accept or reject the license terms for the application.
    Provides YES and NO buttons to confirm or decline the agreement.

    Args:
        *args: Additional positional arguments for QDialog.
        **kwargs: Additional keyword arguments for QDialog.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"Accept {APP_NAME_LONG} License")
        self.setModal(True)

        layout = QVBoxLayout()

        layout.addWidget(
            QLabel(f"I have read and I agree to the {APP_NAME} license terms."),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        button_layout = QHBoxLayout()

        yes_button = QPushButton("YES")
        yes_button.clicked.connect(self.accept)

        no_button = QPushButton("NO")
        no_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(yes_button, alignment=Qt.AlignmentFlag.AlignRight)
        button_layout.addWidget(no_button, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addLayout(button_layout)

        self.setLayout(layout)
