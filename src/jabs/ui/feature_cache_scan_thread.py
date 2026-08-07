"""Background scan of a project's feature cache."""

import logging

from PySide6.QtCore import QObject, QThread, Signal, SignalInstance

from jabs.project import Project, scan_project_feature_cache

logger = logging.getLogger(__name__)


class FeatureCacheScanThread(QThread):
    """Scans a project's on-disk feature cache without blocking the GUI.

    The scan only reads cache metadata, but it touches every per-identity cache
    directory in the project, which on a network filesystem is slow enough to be
    worth keeping off the main thread.

    Args:
        project: Project whose feature cache should be scanned.
        parent: Parent object; owns this thread for lifetime purposes.
    """

    # emits (project, dict[str, VideoFeatureCacheStatus]). The project is included
    # so a receiver can discard results that arrive after a different project has
    # been opened.
    scan_complete: SignalInstance = Signal(object, object)

    def __init__(self, project: Project, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._project = project
        self._should_terminate = False

    def request_termination(self) -> None:
        """Request the scan to stop at the next video boundary.

        Partial results are discarded: nothing is emitted once termination has
        been requested. Safe to call from the main Qt GUI thread; assignment to a
        boolean is atomic in CPython, so no additional synchronization is needed
        (the same approach used by the training and classification threads).
        """
        self._should_terminate = True

    def run(self) -> None:
        """Scan the project's feature cache and emit the per-video status."""
        try:
            statuses = scan_project_feature_cache(
                self._project, should_continue=lambda: not self._should_terminate
            )
        except Exception:
            # A failed scan only costs the cache status display, so log it and
            # leave the status unknown rather than surfacing an error dialog.
            logger.exception("Feature cache scan failed")
            return

        if self._should_terminate:
            logger.debug("Feature cache scan terminated before completing")
            return
        self.scan_complete.emit(self._project, statuses)
