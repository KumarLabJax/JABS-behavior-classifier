import json
import typing

import jabs.feature_extraction as feature_extraction
from jabs.version import version_str

if typing.TYPE_CHECKING:
    from .project_paths import ProjectPaths


class SettingsManager:
    """
    Class to manage project properties/settings.
    """

    def __init__(self, project_paths: "ProjectPaths"):
        """
        Initialize the ProjectProperties.
        :param project_paths: ProjectPaths object to manage file paths.
        """
        self._paths = project_paths
        self._project_info = self._load_project_file()

    def _load_project_file(self) -> dict:
        """
        Load project properties from the project file.
        :return: Dictionary of project properties.
        """
        try:
            with self._paths.project_file.open(mode="r", newline="\n") as f:
                settings = json.load(f)
        except FileNotFoundError:
            settings = {}

        # Ensure default keys exist
        settings.setdefault("behavior", {})
        settings.setdefault("window_sizes", [feature_extraction.DEFAULT_WINDOW_SIZE])

        return settings

    def save_project_file(self, data: dict | None = None):
        """
        Save project properties & settings to the project file.
        :param data: Dictionary with state information to save.
        """
        # Merge data with current metadata
        if data is not None:
            self._project_info.update(data)

        self._project_info["version"] = version_str()

        # Save combined info to file
        with self._paths.project_file.open(mode="w", newline="\n") as f:
            json.dump(self._project_info, f, indent=2, sort_keys=True)

    @property
    def project_settings(self) -> dict:
        """
        Get a copy of the current project properties and settings.
        :return: dict
        """
        return dict(self._project_info)

    def save_behavior(self, behavior: str, data: dict):
        """
        Save a behavior to project file.
        :param behavior: Behavior name.
        :param data: Dictionary of behavior settings.
        """

        defaults = self._project_info.get("defaults", {})

        all_behavior_data = self._project_info.get("behavior", {})
        merged_data = all_behavior_data.get(behavior, defaults)
        merged_data.update(data)

        all_behavior_data[behavior] = merged_data
        self.save_project_file({"behavior": all_behavior_data})

    def get_behavior(self, behavior: str) -> dict:
        """
        Get metadata specific to a requested behavior.
        :param behavior: Behavior key to read.
        :return: Dictionary of behavior metadata.
        """
        return self._project_info.get("behavior", {}).get(behavior, {})

    def remove_behavior(self, behavior: str) -> None:
        # remove from project settings
        try:
            del self._project_info["behavior"][behavior]
            self.save_project_file()
        except KeyError:
            pass

    def update_version(self):
        """
        Update the version number in the metadata if it differs from the current version.
        """
        current_version = self._project_info.get("version")
        if current_version != version_str():
            self.save_project_file({"version": version_str()})
