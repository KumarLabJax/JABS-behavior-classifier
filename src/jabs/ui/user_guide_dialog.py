from typing import Any

import markdown2
from PySide6 import QtCore
from PySide6.QtCore import Qt, QUrl
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QDialog,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from jabs.resources import DOCS_DIR


class UserGuidePage(QWebEnginePage):
    """Custom web engine page that intercepts markdown link navigation.

    This page prevents the browser from directly navigating to markdown files
    and instead signals the parent dialog to load and render them properly.
    """

    def __init__(self, parent: "UserGuideDialog") -> None:
        """Initialize the custom page.

        Args:
            parent: The UserGuideDialog that owns this page.
        """
        super().__init__(parent)
        self._dialog = parent

    def acceptNavigationRequest(
        self, url: QUrl, nav_type: QWebEnginePage.NavigationType, is_main_frame: bool
    ) -> bool:
        """Intercept navigation requests to handle internal markdown links.

        Args:
            url: The URL being navigated to.
            nav_type: The type of navigation (link click, form submit, etc).
            is_main_frame: Whether this is the main frame or an iframe.

        Returns:
            True if navigation should proceed normally, False if we handle it custom.
        """
        # Only intercept link clicks in the main frame
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked and is_main_frame:
            path = url.path()

            # Check if this is a link to a markdown file (internal documentation link)
            # Look for .md extension in the path
            if path and (".md" in path):
                # Extract just the filename, removing any directory path
                filename = path.split("/")[-1]

                # Add back the fragment (anchor) if present
                if url.hasFragment():
                    filename = f"{filename}#{url.fragment()}"

                # Load through our custom rendering, deferred to avoid reentrancy
                # Use QTimer.singleShot to defer the call until after navigation handling completes
                QtCore.QTimer.singleShot(0, lambda: self._dialog._load_content_from_path(filename))
                return False  # Prevent default navigation

        # Allow all other navigation (initial load, etc.)
        return True


class UserGuideDialog(QDialog):
    """Dialog that displays the JABS user guide with tree navigation.

    The dialog renders markdown documentation files as HTML in a web view,
    with a hierarchical tree navigation panel for browsing different topics.
    """

    def __init__(self, app_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the user guide dialog.

        Args:
            app_name: The application name to display in the window title.
            *args: Variable length argument list passed to QDialog.
            **kwargs: Arbitrary keyword arguments passed to QDialog.
        """
        super().__init__(*args, **kwargs)
        self.setWindowTitle(f"{app_name} User Guide")
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.setWindowFlag(QtCore.Qt.WindowType.Tool)
        self.resize(1200, 700)

        # Create tree widget for navigation
        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("Topics")
        self._tree.setMinimumWidth(250)
        self._tree.setMaximumWidth(350)
        self._tree.itemClicked.connect(self._on_tree_item_clicked)

        # Create web view for content with custom page for link handling
        self._web_engine_view = QWebEngineView()
        self._web_engine_view.setPage(UserGuidePage(self))

        # Create splitter to hold tree and content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._tree)
        splitter.addWidget(self._web_engine_view)
        splitter.setStretchFactor(0, 0)  # Tree doesn't stretch
        splitter.setStretchFactor(1, 1)  # Content stretches

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(splitter)

        close_button = QPushButton("CLOSE")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.setLayout(layout)

        # Build tree and load initial content
        self._build_tree()
        # self._load_initial_content()
        self._load_content_from_path("overview.md")

    def _build_tree(self) -> None:
        """Build the hierarchical tree structure for documentation navigation.

        Creates a tree widget populated with topics and subtopics from the user guide.
        Each tree item stores a file path (with optional anchor) that is loaded when
        the item is clicked. Parent nodes can be clicked to load section overview pages.
        """
        # Define the tree structure with display name and file path
        tree_structure = {
            "Overview": "overview.md",
            "Project Setup": {
                "_file": "project-setup.md",
                "Project Directory": "project-setup.md#project-directory",
                "Initialization & jabs-init": "project-setup.md#initialization-jabs-init",
                "JABS Directory Structure": "project-setup.md#jabs-directory-structure",
            },
            "GUI": {
                "_file": "gui.md",
                "Main Window": "gui.md#main-window",
                "Classifier Controls": "gui.md#classifier-controls",
                "Timeline Visualizations": "gui.md#timeline-visualizations",
                "Video Controls": "gui.md#video-controls",
                "Menu": "gui.md#menu",
                "Overlays": "gui.md#overlays",
            },
            "Labeling": {
                "_file": "labeling.md",
                "Selecting Frames": "labeling.md#selecting-frames",
                "Applying Labels": "labeling.md#applying-labels",
                "Timeline Annotations": "labeling.md#timeline-annotations",
                "Identity Gaps": "labeling.md#identity-gaps",
                "Keyboard Shortcuts": "labeling.md#keyboard-shortcuts",
            },
            "Command Line Tools": {
                "_file": "cli-tools.md",
                "jabs-classify": "cli-tools.md#jabs-classify",
                "jabs-features": "cli-tools.md#jabs-features",
                "jabs-cli": "cli-tools.md#jabs-cli",
            },
            "File Formats": {
                "_file": "file-formats.md",
                "Prediction File": "file-formats.md#prediction-file",
                "Feature File": "file-formats.md#feature-file",
            },
            "Features Reference": "features.md",
            "Keyboard Shortcuts Reference": "keyboard-shortcuts.md",
        }

        def add_tree_items(
            parent: QTreeWidget | QTreeWidgetItem, structure: dict[str, Any]
        ) -> None:
            """Recursively add tree items from the documentation structure."""
            for name, value in structure.items():
                # Skip internal keys like "_file"
                if name.startswith("_"):
                    continue

                item = QTreeWidgetItem(parent)
                item.setText(0, name)
                if isinstance(value, dict):
                    # This is a parent node with children
                    # Check if there's a _file key for the parent
                    if "_file" in value:
                        item.setData(0, Qt.ItemDataRole.UserRole, value["_file"])
                    add_tree_items(item, value)
                else:
                    # This is a leaf node with a file path
                    item.setData(0, Qt.ItemDataRole.UserRole, value)

        add_tree_items(self._tree, tree_structure)
        self._tree.expandAll()

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Handle tree item click event.

        Loads the documentation content associated with the clicked tree item.

        Args:
            item: The tree widget item that was clicked.
            _column: The column index (unused but required by Qt signal).
        """
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if file_path:
            self._load_content_from_path(file_path)

    def _load_initial_content(self) -> None:
        """Load the initial content displayed when the dialog opens.

        Loads the overview page as the default starting content.
        """
        self._load_content_from_path("overview.md")

    def _load_content_from_path(self, path_str: str) -> None:
        """Load and render markdown content from a documentation file.

        Converts the markdown file to HTML with styling and displays it in the web view.
        Supports anchor links for scrolling to specific sections within a page.

        Args:
            path_str: Relative path to the markdown file within the user_guide directory.
                     May include an anchor (e.g., "gui.md#main-window").
        """
        # Split anchor if present
        parts = path_str.split("#")
        file_part = parts[0]
        anchor = f"#{parts[1]}" if len(parts) > 1 else ""

        user_guide_path = DOCS_DIR / "user_guide" / file_part

        # need to specify a base URL when displaying the html content due to
        # the relative img urls in the markdown documents
        # Use the user_guide directory as the base to allow relative paths to work
        base_url = QUrl(f"{(DOCS_DIR / 'user_guide').as_uri()}/")

        def error_html(message: str) -> str:
            """Generate HTML error page with the given error message."""
            return f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; background: #f9f9f9; color: #333; padding: 20px; }}
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

            # Wrap content with basic styling and anchor scrolling
            scroll_script = ""
            if anchor:
                scroll_script = f"""
                <script>
                    window.addEventListener('load', function() {{
                        const target = document.querySelector('{anchor}');
                        if (target) {{
                            target.scrollIntoView({{ behavior: 'smooth' }});
                        }}
                    }});
                </script>
                """

            html = f"""
                <html>
                <head>
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                            line-height: 1.6;
                            color: #333;
                            max-width: 900px;
                            margin: 0 auto;
                            padding: 20px;
                            background: #fff;
                        }}
                        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        h2 {{ color: #34495e; margin-top: 24px; }}
                        h3 {{ color: #34495e; }}
                        code {{
                            background: #f4f4f4;
                            padding: 2px 6px;
                            border-radius: 3px;
                            font-family: "Courier New", Courier, monospace;
                        }}
                        pre {{
                            background: #f4f4f4;
                            padding: 12px;
                            border-radius: 5px;
                            overflow-x: auto;
                        }}
                        pre code {{
                            background: none;
                            padding: 0;
                        }}
                        table {{
                            border-collapse: collapse;
                            width: 100%;
                            margin: 20px 0;
                        }}
                        th, td {{
                            border: 1px solid #ddd;
                            padding: 12px;
                            text-align: left;
                        }}
                        th {{
                            background-color: #3498db;
                            color: white;
                        }}
                        tr:nth-child(even) {{
                            background-color: #f9f9f9;
                        }}
                        img {{
                            max-width: 100%;
                            height: auto;
                        }}
                        blockquote {{
                            border-left: 4px solid #3498db;
                            padding-left: 16px;
                            margin-left: 0;
                            color: #666;
                        }}
                    </style>
                </head>
                <body>
                    {html}
                    {scroll_script}
                </body>
                </html>
            """

        except OSError as e:
            html = error_html(
                f"<h2>Unable to read the user guide file.</h2><p><small>{e}</small></p>"
            )
        except UnicodeDecodeError:
            html = error_html("<h2>Unable to decode file (invalid encoding).</h2>")
        except markdown2.MarkdownError as e:
            html = error_html(f"<h2>Markdown parsing error.</h2><p><small>{e}</small></p>")

        self._web_engine_view.setHtml(html, baseUrl=base_url)
