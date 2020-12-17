from PySide2.QtWidgets import QVBoxLayout, QDialog, QLabel, QPushButton
from PySide2.QtCore import Qt

from src.version import version_str


class AboutDialog(QDialog):
    """ dialog that shows application info such as version and copyright """

    def __init__(self, app_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"About {app_name}")

        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"Version: {version_str()}"),
            alignment=Qt.AlignCenter)

        label = QLabel(
            f"{app_name} developed by the "
            "<a href='https://www.jax.org/research-and-faculty/research-labs/the-kumar-lab'>Kumar Lab</a> "
            "at The Jackson Laboratory")
        label.setOpenExternalLinks(True)

        layout.addWidget(label, alignment=Qt.AlignCenter)
        layout.addWidget(QLabel(
            "Copyright 2020 The Jackson Laboratory. All Rights Reserved"),
            alignment=Qt.AlignCenter)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.close)

        layout.addWidget(ok_button, alignment=Qt.AlignLeft)

        self.setLayout(layout)
