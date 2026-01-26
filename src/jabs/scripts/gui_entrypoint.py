import argparse
import os
import sys

# suppress some potential harmless warnings from Chromium when user opens UserGuideDialog on some platforms
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
    "--disable-skia-graphite --disable-logging --log-level=3"
)
os.environ["QT_LOGGING_RULES"] = "qt.webenginecontext=false"
from PySide6 import QtWidgets
from PySide6.QtGui import QIcon

from jabs.core.constants import APP_NAME, APP_NAME_LONG, ORG_NAME
from jabs.resources import ICON_PATH
from jabs.ui import MainWindow
from jabs.version import version_str


def main():
    """main entrypoint for JABS video labeling and classifier GUI

    takes one optional positional argument: path to project directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "project_dir", nargs="?", help="Path to JABS project directory to open on startup"
    )
    parser.add_argument("--version", action="version", version=f"JABS {version_str()}")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setWindowIcon(QIcon(str(ICON_PATH)))

    main_window = MainWindow(app_name=APP_NAME, app_name_long=APP_NAME_LONG)
    main_window.show()
    if main_window.show_license_dialog() != QtWidgets.QDialog.DialogCode.Accepted:
        sys.exit(1)

    if args.project_dir is not None:
        # this forces the GUI to process events before opening the project
        # this is necessary to avoid a race condition where the main window
        # is not fully initialized before trying to open the project
        QtWidgets.QApplication.processEvents()
        try:
            main_window.open_project(args.project_dir)
        except Exception as e:
            sys.exit(f"Error opening project:  {e}")

    # user accepted license terms, run the main application loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
