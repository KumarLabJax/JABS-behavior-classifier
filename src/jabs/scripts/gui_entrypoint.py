import argparse
import contextlib
import logging
import multiprocessing
import os
import sys

# PERFORMANCE FIX: Use fork instead of spawn for faster process creation on macOS
#
# Background: Initializing JABS-AppProcessPool on macOS was taking significant time, which
# got significantly worse (25s)  with macOS Tahoe (maybe Sequoia+ ?)
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
#   1. Initialize pool BEFORE Qt initializes (forked from single-threaded state)
#   2. Workers only read files and do data processing (no Qt usage)
#   3. Extensive testing shows stability in practice
#
# TODO: Test Windows to see if there is benefit to using "fork" there as well.
if sys.platform == "darwin":
    # try to use 'fork' start method on macOS, suppress RuntimeError if it fails -- we'll fall back to default
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("fork", force=True)

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

# Set log level from environment variable if present
log_level_str = os.environ.get("JABS_LOG_LEVEL", "WARNING").upper()
try:
    log_level = getattr(logging, log_level_str)
except AttributeError:
    log_level = logging.WARNING
    logger = logging.getLogger("jabs.gui_entrypoint")
    logger.warning(f"Invalid JABS_LOG_LEVEL '{log_level_str}', defaulting to WARNING.")
logging.basicConfig(level=log_level)
logger = logging.getLogger("jabs.gui_entrypoint")


# logger wasn't setup when we set the multiprocessing start method
# if we need to log anything related to that, do it here
if sys.platform == "darwin" and multiprocessing.get_start_method() != "fork":
    logger.warning(
        "Failed to set multiprocessing start method to 'fork' on macOS, "
        "this may lead to slower process pool initialization."
    )


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
    # QApplication creates threads; forking after that can be unsafe
    logger.info("Initializing process pool (before Qt)...")
    logger.debug(f"multiprocessing start method: '{multiprocessing.get_start_method()}'")
    process_pool = ProcessPoolManager(name="JABS-AppProcessPool")
    if multiprocessing.get_start_method() == "fork":
        # on fork platforms, start the pool and wait for workers to be ready
        process_pool.warm_up(wait=True)
    else:
        # on non-fork platforms, start the pool without waiting, workers will be spawned on-demand
        process_pool.warm_up(wait=False)
    logger.info(f"Process pool ready ({process_pool.max_workers} workers)")

    # Now safe to create QApplication (fork already happened)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setWindowIcon(QIcon(str(ICON_PATH)))

    main_window = MainWindow(
        app_name=APP_NAME, app_name_long=APP_NAME_LONG, process_pool=process_pool
    )
    main_window.show()

    # prompt user to accept license terms (main_window won't show the dialog again if already accepted)
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
