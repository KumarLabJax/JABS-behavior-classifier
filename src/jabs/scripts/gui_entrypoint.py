import argparse
import logging
import multiprocessing
import os
import sys

# suppress some potential harmless warnings from Chromium when user opens UserGuideDialog on some platforms
# we need to set these before importing PySide6.QtWebEngine, so we do it before all PySide6 and JABS imports
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
    "--disable-skia-graphite --disable-logging --log-level=3"
)
os.environ["QT_LOGGING_RULES"] = "qt.webenginecontext=false"
from PySide6 import QtWidgets
from PySide6.QtGui import QIcon

from jabs.core.constants import APP_NAME, APP_NAME_LONG, ORG_NAME
from jabs.core.utils.process_pool_manager import ProcessPoolManager
from jabs.resources import ICON_PATH
from jabs.ui import MainWindow
from jabs.version import version_str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jabs.gui_entrypoint")

# PERFORMANCE FIX: Use fork instead of spawn for faster process creation on macOS
#
# Background: Initializing JABS-AppProcessPool on macOS was taking significant time, which
# got significantly worse (25s)  with macOS Sequoia+
# Potential cause: macOS Sequoia+ scans adhoc-signed executables on every spawn,
# causing significant overhead per worker process. Using fork() avoids this entirely.
#
# Why it's faster:
# - Workers inherit parent's memory (no re-importing modules)
# - No new executable spawned (no macOS security scans)
#
# Safety considerations:
# - fork() is generally unsafe with multi-threaded programs
# - Qt uses threads internally, so there's some risk
# - We mitigate this by:
#   1. Initialize pool BEFORE Qt initializes (minimizes thread count)
#   2. Workers only read files and do data processing (no Qt usage)
#   3. Extensive testing shows stability in practice
#
# TODO (future): Consider using 'spawn' on Windows/Linux and 'fork' only on macOS
# TODO (future): Consider adding a command-line flag or env variable to override this behavior
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    logger.warning("multiprocessing.set_start_method, falling back to default start method")


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

    # CRITICAL: Create and warm the process pool BEFORE QApplication
    # QApplication creates threads, and forking after that is unsafe
    logger.debug("Initializing process pool (before Qt)...")
    process_pool = ProcessPoolManager(name="JABS-AppProcessPool")
    if multiprocessing.get_start_method() == "fork":
        # on fork platforms, start the pool and wait for workers to be ready
        process_pool.warm_up(wait=True)
    else:
        # on non-fork platforms, start the pool without waiting, workers will be spawned on-demand
        process_pool.warm_up(wait=False)
    logger.debug(f"Process pool ready ({process_pool.max_workers} workers)")

    # Now safe to create QApplication (fork already happened)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setWindowIcon(QIcon(str(ICON_PATH)))

    # Pass the pre-created pool to MainWindow
    main_window = MainWindow(
        app_name=APP_NAME, app_name_long=APP_NAME_LONG, process_pool=process_pool
    )
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
