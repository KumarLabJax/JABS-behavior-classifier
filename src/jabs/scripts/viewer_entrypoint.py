"""GUI entry point for jabs-viewer, a standalone player for videos and pose files.

Qt and ``jabs.ui`` imports are deferred into :func:`main` so the QtWebEngine
environment flags can be set before anything imports ``jabs.ui`` (which pulls in QtWebEngine).
"""

import argparse
import os
import sys
from pathlib import Path

# stdlib-only module, so importing it keeps this module's scope free of Qt
from jabs.scripts.gui_entrypoint import _configure_logging

VIEWER_APP_NAME = "JABS Viewer"


def main() -> None:
    """Main entry point for jabs-viewer.

    Takes one optional positional argument: path to a video or pose file to open.
    """
    _configure_logging()

    # Must be set before importing anything that pulls in QtWebEngine (jabs.ui).
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--disable-skia-graphite --disable-logging --log-level=3"
    )
    os.environ["QT_LOGGING_RULES"] = "qt.webenginecontext=false"
    from PySide6 import QtWidgets
    from PySide6.QtGui import QIcon

    from jabs.core.constants import ORG_NAME
    from jabs.resources import ICON_PATH
    from jabs.ui.viewer import ViewerWindow
    from jabs.version import version_str

    parser = argparse.ArgumentParser(
        description="View a video, or a JABS pose file on its own, with pose and "
        "segmentation overlays."
    )
    parser.add_argument(
        "file", nargs="?", type=Path, help="Path to a video or JABS pose (.h5) file to open"
    )
    parser.add_argument("--version", action="version", version=f"JABS {version_str()}")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(VIEWER_APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = ViewerWindow(app_name=VIEWER_APP_NAME)
    window.resize(1024, 900)
    window.show()

    if args.file is not None:
        # let the window finish showing before loading the file
        QtWidgets.QApplication.processEvents()
        window.open_file(args.file)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
