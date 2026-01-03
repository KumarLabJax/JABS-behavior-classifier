"""Dialog for displaying training reports as HTML."""

from PySide6.QtCore import QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class TrainingReportDialog(QDialog):
    """Dialog for displaying training report HTML content."""

    def __init__(self, html_content: str, title: str = "Training Report", parent=None):
        """Initialize the training report dialog.

        Args:
            html_content: HTML content to display
            title: Window title
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 700)

        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create web view
        self.web_view = QWebEngineView()
        self.web_view.setHtml(html_content)

        main_layout.addWidget(self.web_view)

        # Create bottom row with info text and close button
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(10, 5, 10, 5)

        info_label = QLabel("Report saved in jabs/training_logs directory")
        bottom_layout.addWidget(info_label)

        bottom_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom_layout.addWidget(close_button)

        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

        # Add keyboard shortcut for Cmd+Q (macOS) / Ctrl+Q to quit application
        quit_shortcut = QShortcut(QKeySequence.StandardKey.Quit, self)
        quit_shortcut.activated.connect(self._quit_application)

    def _quit_application(self):
        """Quit the application.

        Uses QTimer.singleShot to defer the quit operation until after
        the dialog's modal event loop finishes (exec() is blocking).
        """
        self.close()
        QTimer.singleShot(0, QApplication.instance().quit)
