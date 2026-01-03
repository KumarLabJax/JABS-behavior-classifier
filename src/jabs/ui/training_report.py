"""Dialog for displaying training reports as HTML."""

from textwrap import dedent

import markdown2
from PySide6.QtGui import QIcon
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
)
from qt_material_icons import MaterialIcon


class TrainingReportDialog(QDialog):
    """Dialog for displaying training report content."""

    def __init__(self, markdown_content: str, title: str = "Training Report", parent=None):
        """Initialize the training report dialog.

        Args:
            markdown_content: Markdown-formatted training report content
            title: Window title
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 700)

        # Store markdown content for copying to clipboard
        self._markdown_content = markdown_content

        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Convert markdown to HTML and create web view
        html_content = self._markdown_to_html(markdown_content)
        self.web_view = QWebEngineView()
        self.web_view.setHtml(html_content)

        main_layout.addWidget(self.web_view)

        # Create bottom row with info text and copy button
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(10, 5, 10, 5)

        info_label = QLabel("Report saved in jabs/training_logs directory")
        bottom_layout.addWidget(info_label)

        bottom_layout.addStretch()

        # Create icon button for copying markdown to clipboard
        copy_button = QToolButton()
        copy_button.setIcon(QIcon(MaterialIcon("content_copy").pixmap(20)))
        copy_button.setToolTip("Copy Markdown to Clipboard")
        copy_button.clicked.connect(self._copy_markdown_to_clipboard)
        bottom_layout.addWidget(copy_button)

        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

    def _copy_markdown_to_clipboard(self):
        """Copy the markdown content to the system clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self._markdown_content)

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown text to HTML.

        Args:
            markdown_text: Markdown-formatted string

        Returns:
            HTML string with basic styling
        """
        html_content = markdown2.markdown(markdown_text, extras=["tables", "fenced-code-blocks"])

        # Wrap in basic HTML document with styling
        html = dedent(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
                        line-height: 1.6;
                        max-width: 1200px;
                        margin: 20px auto;
                        padding: 0 20px;
                        color: #333;
                    }}
                    h1 {{
                        border-bottom: 2px solid #333;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        border-bottom: 1px solid #ccc;
                        padding-bottom: 8px;
                        margin-top: 30px;
                    }}
                    h3 {{
                        margin-top: 20px;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
                    code {{
                        background-color: #f4f4f4;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }}
                    ul {{
                        line-height: 1.8;
                    }}
                </style>
            </head>
            <body>
            {html_content}
            </body>
            </html>
        """).strip()
        return html
