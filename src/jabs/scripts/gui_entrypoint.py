"""GUI entry point for JABS.

All non-stdlib imports are deferred into :func:`main` rather than performed at
module scope. On macOS the process pool uses the ``forkserver`` start method (see
:func:`_select_start_method`): its server process imports this module to locate
worker functions but never runs :func:`main`, so keeping the imports out of module
scope keeps that server free of Qt/Foundation. Otherwise the workers it forks
would abort via the Objective-C fork-safety guard (surfacing as
``BrokenProcessPool``) on their first Accelerate or Qt call.
"""

import argparse
import contextlib
import logging
import multiprocessing
import os
import sys

logger = logging.getLogger("jabs.gui_entrypoint")


def _configure_logging() -> None:
    """Configure root logging from the ``JABS_LOG_LEVEL`` env var (default WARNING)."""
    log_level_str = os.environ.get("JABS_LOG_LEVEL", "WARNING").upper()
    log_level = getattr(logging, log_level_str, None)
    if not isinstance(log_level, int):
        logging.basicConfig(level=logging.WARNING)
        logger.warning("Invalid JABS_LOG_LEVEL '%s', defaulting to WARNING.", log_level_str)
        return
    logging.basicConfig(level=log_level)


def _select_start_method() -> None:
    """Select the multiprocessing start method (macOS only).

    macOS: use ``forkserver``. ``fork`` is unsafe here -- forked workers abort
    via the Objective-C fork-safety guard when they call into Apple Accelerate
    (numpy/scipy) or Qt/Foundation, surfacing as ``BrokenProcessPool`` during
    feature generation. ``spawn`` is safe but cold-starts a fresh interpreter
    per worker (~15-20s on first project load). ``forkserver`` forks workers
    from a single pre-warmed, Qt/Accelerate-free server: fast like ``fork``
    and safe like ``spawn``. See KLAUS-525.

    Other platforms keep their default (Linux fork/forkserver, Windows spawn).

    Configure logging before calling this: it warns (rather than failing) if the
    method could not be set, so the degraded state is at least visible.
    """
    if sys.platform != "darwin":
        return
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("forkserver", force=True)
    method = multiprocessing.get_start_method()
    if method != "forkserver":
        # The macOS default is 'spawn' (safe but slow); 'fork' would be unstable.
        # Warn rather than abort -- spawn still works, just with a slow first load.
        logger.warning(
            "Could not set multiprocessing start method to 'forkserver' (got '%s'); "
            "worker startup will be slower, and 'fork' would risk the worker-pool crash.",
            method,
        )


def main() -> None:
    """Main entry point for the JABS video labeling and classifier GUI.

    Takes one optional positional argument: path to a project directory to open.
    """
    _configure_logging()
    _select_start_method()

    # Deferred imports (see module docstring). The QtWebEngine flags must be set
    # before importing anything that pulls in QtWebEngine (jabs.ui), so these
    # os.environ lines stay above the imports below.
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--disable-skia-graphite --disable-logging --log-level=3"
    )
    os.environ["QT_LOGGING_RULES"] = "qt.webenginecontext=false"
    from PySide6 import QtWidgets
    from PySide6.QtGui import QIcon

    from jabs.core.constants import APP_NAME, APP_NAME_LONG, ORG_NAME
    from jabs.core.utils.process_pool_manager import ProcessPoolManager
    from jabs.project.parallel_workers import preload_worker_modules
    from jabs.resources import ICON_PATH
    from jabs.ui import MainWindow
    from jabs.version import version_str

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "project_dir", nargs="?", help="Path to JABS project directory to open on startup"
    )
    parser.add_argument("--version", action="version", version=f"JABS {version_str()}")
    args = parser.parse_args()

    # Warm the pool up front on fork/forkserver so worker start-up (and the
    # initializer's module preloading) is paid once at launch, making the first
    # project load / training run instant. Leave spawn (e.g. Windows) lazy so it
    # doesn't slow GUI launch -- those workers start on first use instead.
    start_method = multiprocessing.get_start_method()
    logger.info("Initializing process pool (start method: '%s')...", start_method)
    process_pool = ProcessPoolManager(
        name="JABS-AppProcessPool", initializer=preload_worker_modules
    )
    process_pool.warm_up(wait=start_method in ("fork", "forkserver"))
    logger.info("Process pool ready (%d workers)", process_pool.max_workers)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setWindowIcon(QIcon(str(ICON_PATH)))

    main_window = MainWindow(
        app_name=APP_NAME, app_name_long=APP_NAME_LONG, process_pool=process_pool
    )
    main_window.show()

    if args.project_dir is not None:
        # force the GUI to process events before opening the project to avoid a
        # race where the main window is not fully initialized before opening
        QtWidgets.QApplication.processEvents()
        try:
            main_window.open_project(args.project_dir)
        except Exception as e:
            sys.exit(f"Error opening project:  {e}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
