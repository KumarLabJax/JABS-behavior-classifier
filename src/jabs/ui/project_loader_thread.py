import logging

from PySide6.QtCore import QThread, Signal, SignalInstance

from jabs.core.utils.process_pool_manager import ProcessPoolManager
from jabs.project import Project

logger = logging.getLogger(__name__)


class ProjectLoaderThread(QThread):
    """JABS Project Loader Thread

    This thread is used to load a JABS project in the background so that the main
    GUI thread remains responsive. After opening the project it converts the
    project's legacy pose files into the conversion cache
    (:meth:`Project.prepare_poses`), so the GUI never converts a pose on first
    load; a refused file marks its video unusable. It emits signals for
    conversion progress and when the project is loaded.
    """

    project_loaded: SignalInstance = Signal()
    load_error: SignalInstance = Signal(Exception)
    # (completed, total) legacy pose files converted
    conversion_progress: SignalInstance = Signal(int, int)

    def __init__(
        self,
        project_path: str,
        process_pool: ProcessPoolManager | None = None,
        parent=None,
        session_tracking_enabled: bool = False,
    ):
        super().__init__(parent)
        self._project_path = project_path
        self._process_pool = process_pool
        self._project = None
        self._tracking_enabled = session_tracking_enabled

    def run(self):
        """Run the thread."""
        # Open the project, this can take a while
        try:
            self._project = Project(
                self._project_path,
                process_pool=self._process_pool,
                enable_session_tracker=self._tracking_enabled,
            )
        except Exception as e:
            # if there was an exception, we'll emit the Exception as a signal so that
            # the main GUI thread can handle it
            self.load_error.emit(e)
            return

        try:
            self._project.prepare_poses(self.conversion_progress.emit)
        except Exception:
            # converting ahead only moves work out of the first pose load; a pose
            # that cannot be converted is still reported when the video is opened
            logger.exception("Converting legacy pose files ahead of use failed")
        self.project_loaded.emit()

    @property
    def project(self):
        """Return the loaded project."""
        return self._project
