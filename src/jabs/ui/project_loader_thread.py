from PySide6.QtCore import QThread, Signal, SignalInstance

from jabs.project import Project


class ProjectLoaderThread(QThread):
    """JABS Project Loader Thread

    This thread is used to load a JABS project in the background so that the main
    GUI thread remains responsive. It emits signals when the project is loaded.
    """

    project_loaded: SignalInstance = Signal()
    load_error: SignalInstance = Signal(Exception)

    def __init__(self, project_path: str, parent=None, session_tracking_enabled: bool = False):
        super().__init__(parent)
        self._project_path = project_path
        self._project = None
        self._tracking_enabled = session_tracking_enabled

    def run(self):
        """Run the thread."""
        # Open the project, this can take a while
        try:
            self._project = Project(
                self._project_path, enable_session_tracker=self._tracking_enabled
            )
            self.project_loaded.emit()
        except Exception as e:
            # if there was an exception, we'll emit the Exception as a signal so that
            # the main GUI thread can handle it
            self.load_error.emit(e)

    @property
    def project(self):
        """Return the loaded project."""
        return self._project
