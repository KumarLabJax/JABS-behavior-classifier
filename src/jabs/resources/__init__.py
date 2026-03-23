"""Resource file paths.

This package provides package-aware access to application resources
such as documentation files and icons using `importlib.resources`.

Attributes:
    DOCS_DIR (pathlib.Path): Path object to the documentation directory.
    ICON_PATH (pathlib.Path): Path object to the application icon.
    FAIL_WHALE_PATH (pathlib.Path): Path object to the fail whale error image.
"""

import importlib.resources

_RESOURCES = importlib.resources.files("jabs.resources")

DOCS_DIR = _RESOURCES / "docs"
ICON_PATH = _RESOURCES / "icon.png"
FAIL_WHALE_PATH = _RESOURCES / "fail_whale.png"


__all__ = [
    "DOCS_DIR",
    "FAIL_WHALE_PATH",
    "ICON_PATH",
]
