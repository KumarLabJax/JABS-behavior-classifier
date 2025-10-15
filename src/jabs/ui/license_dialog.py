from importlib import resources

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
)

from ..constants import APP_NAME, APP_NAME_LONG


def _read_license_text() -> str:
    try:
        with (
            resources.files("jabs.resources.docs")
            .joinpath("LICENSE")
            .open("r", encoding="utf-8") as f
        ):
            return f.read()
    except Exception:
        return "LICENSE not found in package."


class LicenseAgreementDialog(QDialog):
    """Dialog for accepting the application license agreement.

    Presents the user with a message to accept or reject the license terms for the application.
    Provides YES and NO buttons to confirm or decline the agreement.

    Args:
        *args: Additional positional arguments for QDialog.
        **kwargs: Additional keyword arguments for QDialog.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{key: val for key, val in kwargs.items() if key != "view_only"})
        self._view_only = kwargs.get("view_only", False)
        self.setWindowTitle(
            f"Accept {APP_NAME_LONG} License"
            if not self._view_only
            else f"{APP_NAME_LONG} License"
        )
        self.setModal(True)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(600, 350)

        layout = QVBoxLayout(self)

        license_view = QPlainTextEdit()
        license_view.setReadOnly(True)
        license_view.setPlainText(_read_license_text())
        license_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(license_view)

        if self._view_only:
            # this is the license viewer, not the acceptance dialog
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            buttons.accepted.connect(self.accept)
        else:
            # this is the acceptance dialog, need to add the prompt and Yes/No buttons
            layout.addWidget(
                QLabel(f"I have read and I agree to the {APP_NAME} license terms:"),
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def sizeHint(self) -> QSize:
        """Preferred starting size; layout can expand beyond this"""
        return QSize(700, 420)
