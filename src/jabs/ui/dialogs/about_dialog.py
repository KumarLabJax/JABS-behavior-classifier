from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from jabs.version import version_str


class AboutDialog(QDialog):
    """dialog that shows application info such as version and copyright"""

    def __init__(self, app_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"About {app_name}")

        layout = QVBoxLayout()

        layout.addWidget(
            QLabel(f"Version: {version_str()}"), alignment=Qt.AlignmentFlag.AlignCenter
        )

        label = QLabel(
            f"{app_name} developed by the "
            "<a href='https://www.jax.org/research-and-faculty/research-labs/the-kumar-lab'>Kumar Lab</a> "
            "at The Jackson Laboratory"
        )
        label.setOpenExternalLinks(True)
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(
            QLabel("Copyright 2025 The Jackson Laboratory. All Rights Reserved"),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        email_label = QLabel("<a href='mailto:jabs@jax.org'>jabs@jax.org</a>")
        email_label.setOpenExternalLinks(True)
        layout.addWidget(email_label, alignment=Qt.AlignmentFlag.AlignCenter)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.close)

        layout.addWidget(ok_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.setLayout(layout)
