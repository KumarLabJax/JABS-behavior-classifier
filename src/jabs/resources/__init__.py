"""Resource file paths.

This package provides package-aware access to application resources
such as documentation files and icons using `importlib.resources`.

Attributes:
    DOCS_DIR (pathlib.Path): Path object to the documentation directory.
    ICON_PATH (pathlib.Path): Path object to the application icon.
"""

import importlib.resources

DOCS_DIR = importlib.resources.files("jabs.resources") / "docs"
ICON_PATH = importlib.resources.files("jabs.resources") / "icon.png"


__all__ = [
    "DOCS_DIR",
    "ICON_PATH",
]
