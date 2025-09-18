import markdown2
from PySide6 import QtCore
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QDialog, QPushButton, QVBoxLayout

from jabs.resources import DOCS_DIR


class UserGuideDialog(QDialog):
    """dialog that shows html rendering of user guide"""

    def __init__(self, app_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"{app_name} User Guide")
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setWindowFlag(QtCore.Qt.WindowType.Tool)
        self.resize(1000, 600)
        self._web_engine_view = QWebEngineView()
        self._load_content()

        layout = QVBoxLayout()

        layout.addWidget(self._web_engine_view)

        close_button = QPushButton("CLOSE")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.setLayout(layout)

    def _load_content(self):
        user_guide_path = DOCS_DIR / "user_guide" / "user_guide.md"

        # need to specify a base URL when displaying the html content due to
        # the relative img urls in the user_guide.md document
        base_url = QUrl(f"{user_guide_path.parent.as_uri()}/")

        def error_html(message):
            return f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; background: #f9f9f9; color: #333; }}
                        .error-container {{
                            margin: 40px auto;
                            padding: 32px 40px;
                            max-width: 600px;
                            background: #fff3f3;
                            border: 1px solid #e0b4b4;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                        }}
                        h1 {{ color: #b22222; margin-bottom: 0.5em; }}
                        h2 {{ color: #b22222; margin-top: 0; font-size: 1.2em; }}
                        p {{ margin-top: 0.5em; }}
                    </style>
                </head>
                <body>
                    <div class="error-container">
                        <h1>Error Loading User Guide</h1>
                        {message}
                    </div>
                </body>
                </html>
            """

        try:
            html = markdown2.markdown_path(
                str(user_guide_path), extras=["fenced-code-blocks", "header-ids", "tables"]
            )
        except OSError as e:
            html = error_html(
                f"<h2>Unable to read the user guide file.</h2><p><small>{e}</small></p>"
            )
        except UnicodeDecodeError:
            html = error_html("<h2>Unable to decode file (invalid encoding).</h2>")
        except markdown2.MarkdownError as e:
            html = error_html(f"<h2>Markdown parsing error.</h2><p><small>{e}</small></p>")

        self._web_engine_view.setHtml(html, baseUrl=base_url)
