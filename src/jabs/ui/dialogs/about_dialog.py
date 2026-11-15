"""Dialog showing application information: version, copyright, and installed extras."""

import html
from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from jabs.core.utils import ExtraStatus, extras_status
from jabs.version import version_str

# Distribution whose optional extras are reported.
_DISTRIBUTION = "jabs-behavior-classifier"

# User guide page that explains the extras and how to install them.
EXTRAS_USER_GUIDE_PAGE = "optional-extras.md"


def extra_status_text(status: ExtraStatus) -> str:
    """Return a short, human-readable install status for one extra.

    Args:
        status: The extra's install status.

    Returns:
        ``"installed"``, ``"not installed"``, or, when only some of its packages are
        present, ``"partially installed (missing: ...)"``.
    """
    if status.installed:
        return "installed"
    if status.partially_installed:
        return f"partially installed (missing: {', '.join(status.missing)})"
    return "not installed"


def _extras_table_html(statuses: Sequence[ExtraStatus]) -> str:
    """Render the extras and their status as a small rich-text table."""
    if not statuses:
        return "<i>Unable to determine installed extras.</i>"
    rows = "".join(
        f"<tr><td><code>{html.escape(status.name)}</code></td>"
        f"<td style='padding-left: 12px'>{html.escape(extra_status_text(status))}</td></tr>"
        for status in statuses
    )
    return f"<table>{rows}</table>"


class AboutDialog(QDialog):
    """Dialog that shows application info such as version, copyright, and installed extras.

    Signals:
        user_guide_page_requested (str): Emitted with a user guide page path when the
            user follows a link to the user guide. The dialog closes itself first, so
            that the guide is not blocked behind this modal dialog.
    """

    user_guide_page_requested = Signal(str)

    def __init__(
        self,
        app_name: str,
        *args,
        extras: Sequence[ExtraStatus] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the About dialog.

        Args:
            app_name: Application name shown in the title and credits.
            *args: Positional arguments passed to QDialog.
            extras: Install status of the optional extras to show. Defaults to the
                extras of the installed JABS distribution.
            **kwargs: Keyword arguments passed to QDialog.
        """
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
            QLabel("Copyright (c) 2023-2026 The Jackson Laboratory"),
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        email_label = QLabel("<a href='mailto:jabs@jax.org'>jabs@jax.org</a>")
        email_label.setOpenExternalLinks(True)
        layout.addWidget(email_label, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(
            self._build_extras_group(extras_status(_DISTRIBUTION) if extras is None else extras)
        )

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.close)

        layout.addWidget(ok_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.setLayout(layout)

    def _build_extras_group(self, extras: Sequence[ExtraStatus]) -> QWidget:
        """Build the box listing each optional extra and whether it is installed."""
        group = QGroupBox("Optional Extras")
        group_layout = QVBoxLayout(group)

        table = QLabel(_extras_table_html(extras))
        table.setTextFormat(Qt.TextFormat.RichText)
        group_layout.addWidget(table)

        guide_link = QLabel(
            f"<a href='{EXTRAS_USER_GUIDE_PAGE}'>How to install optional extras</a>"
        )
        guide_link.setTextFormat(Qt.TextFormat.RichText)
        guide_link.linkActivated.connect(self._on_user_guide_link)
        group_layout.addWidget(guide_link)

        return group

    def _on_user_guide_link(self, page: str) -> None:
        """Close the dialog and ask for the linked user guide page to be shown."""
        self.accept()
        self.user_guide_page_requested.emit(page)
