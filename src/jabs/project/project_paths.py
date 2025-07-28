from pathlib import Path


class ProjectPaths:
    """Class to manage project paths."""

    __JABS_DIR = "jabs"
    __PROJECT_FILE = "project.json"

    def __init__(self, base_path: Path, use_cache: bool = True):
        self._base_path = base_path

        self._jabs_dir = base_path / self.__JABS_DIR
        self._annotations_dir = self._jabs_dir / "annotations"
        self._feature_dir = self._jabs_dir / "features"
        self._prediction_dir = self._jabs_dir / "predictions"
        self._classifier_dir = self._jabs_dir / "classifiers"
        self._archive_dir = self._jabs_dir / "archive"
        self._session_dir = self._jabs_dir / "session"
        self._cache_dir = self._jabs_dir / "cache" if use_cache else None

        self._project_file = self._jabs_dir / self.__PROJECT_FILE

    @property
    def project_dir(self) -> Path:
        """Get the base path of the project."""
        return self._base_path

    @property
    def jabs_dir(self) -> Path:
        """Get the path to the JABS directory."""
        return self._jabs_dir

    @property
    def annotations_dir(self) -> Path:
        """Get the path to the annotations directory."""
        return self._annotations_dir

    @property
    def feature_dir(self) -> Path:
        """Get the path to the features directory."""
        return self._feature_dir

    @property
    def prediction_dir(self) -> Path:
        """Get the path to the predictions directory."""
        return self._prediction_dir

    @property
    def project_file(self) -> Path:
        """Get the path to the project file."""
        return self._project_file

    @property
    def classifier_dir(self) -> Path:
        """Get the path to the classifiers directory."""
        return self._classifier_dir

    @property
    def archive_dir(self) -> Path:
        """Get the path to the archive directory."""
        return self._archive_dir

    @property
    def cache_dir(self) -> Path | None:
        """Get the path to the cache directory."""
        return self._cache_dir

    @property
    def session_dir(self) -> Path:
        """Get the path to the session directory."""
        return self._session_dir

    def create_directories(self):
        """Create all necessary directories for the project."""
        self._annotations_dir.mkdir(parents=True, exist_ok=True)
        self._feature_dir.mkdir(parents=True, exist_ok=True)
        self._prediction_dir.mkdir(parents=True, exist_ok=True)
        self._classifier_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir.mkdir(parents=True, exist_ok=True)

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
