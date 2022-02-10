import os
from pathlib import Path

from PySide2.QtWidgets import QVBoxLayout, QDialog, QPushButton
from PySide2.QtWebEngineWidgets import QWebEngineView
from PySide2.QtCore import Qt, QUrl
import markdown2


class UserGuideDialog(QDialog):
    """ dialog that shows html rendering of user guide """

    _doc_dir = Path(os.path.realpath(__file__)).parent.parent.parent / 'docs'

    def __init__(self, app_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"{app_name} User Guide")
        self.resize(1000, 600)
        self._web_engine_view = QWebEngineView()
        self._load_content()

        layout = QVBoxLayout()

        layout.addWidget(self._web_engine_view)

        close_button = QPushButton("CLOSE")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=Qt.AlignLeft)

        self.setLayout(layout)

    def _load_content(self):

        user_guide_path = self._doc_dir / 'user_guide' / 'user_guide.md'

        # need to specify a base URL when displaying the html content due to
        # the relative img urls in the user_guide.md document
        base_url = QUrl(f'file://{user_guide_path.parent}/')

        try:
            html = markdown2.markdown_path(user_guide_path,  extras=['fenced-code-blocks'])
        except:
            # if there is any error rendering the markdown as html, display
            # an error message instead
            html = '<b>Error Loading User Guide</b>'
        self._web_engine_view.setHtml(html, baseUrl=base_url)
