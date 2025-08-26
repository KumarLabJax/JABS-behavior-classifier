import json
import typing

import jabs.feature_extraction as feature_extraction
from jabs.version import version_str

if typing.TYPE_CHECKING:
    from .project_paths import ProjectPaths


class SettingsManager:
    """Class to manage project properties/settings."""

    def __init__(self, project_paths: "ProjectPaths"):
        """Initialize the ProjectProperties.

        Args:
            project_paths: ProjectPaths object to manage file paths.
        """
        self._paths = project_paths
        self._project_info = self._load_project_file()

    def _load_project_file(self) -> dict:
        """Load project properties from the project file.

        Returns:
            Dictionary of project properties.
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
        """Save project properties & settings to the project file.

        Args:
            data: Dictionary with state information to save.
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
        """Get a copy of the current project properties and settings.

        Returns:
            dict
        """
        return dict(self._project_info)

    @property
    def behavior_names(self) -> list[str]:
        """Get a list of all behaviors defined in the project settings.

        Returns:
            List of behavior names.
        """
        return list(self._project_info.get("behavior", {}).keys())

    @property
    def project_metadata(self) -> dict:
        """Get project-level metadata.

        Returns:
            Dictionary of project-level metadata, or empty dict if none exists.
        """
        return self._project_info.get("metadata", {})

    def video_metadata(self, video: str) -> dict:
        """Get metadata for a specific video.

        Args:
            video: Name of the video file.

        Returns:
            Dictionary of metadata for the specified video, or empty dict if none exists.

        Raises:
            KeyError: If the specified video is not found in the project.
        """
        return self._project_info["video_files"][video].get("metadata", {})

    def set_project_metadata(self, metadata: dict) -> None:
        """Set or replace project and per-video metadata.

        Removes any existing project-level and per-video metadata, then sets new metadata as provided.
        Only the "metadata" field is updated for each video; other fields are preserved.

        Args:
            metadata (dict): Dictionary containing new project-level metadata under the "metadata" key,
                and per-video metadata under the "videos" key. Example:
                {
                    "metadata": {...},
                    "videos": {
                        "video1": {"metadata": {...}},
                        ...
                    }
                }

        Raises:
            KeyError: If metadata for a video is provided for a video not present in the project.
        """
        # Remove existing project-level metadata
        self._project_info.pop("metadata", None)

        # Remove existing metadata from each video
        video_files = self._project_info.get("video_files", {})
        for video_entry in video_files.values():
            video_entry.pop("metadata", None)

        # Set new project-level metadata if provided
        if "metadata" in metadata:
            self._project_info["metadata"] = metadata["metadata"]

        # Update per-video metadata
        for video_name, video_data in metadata.get("videos", {}).items():
            # get the existing video entry, raise KeyError if not found
            video_entry = video_files[video_name]
            video_metadata = video_data.get("metadata")
            if video_metadata:
                video_entry["metadata"] = video_metadata
            video_files[video_name] = video_entry

        # Save changes
        self.save_project_file()

    def save_behavior(self, behavior: str, data: dict):
        """Save a behavior to project file.

        Args:
            behavior: Behavior name.
            data: Dictionary of behavior settings.
        """
        defaults = self._project_info.get("defaults", {})

        all_behavior_data = self._project_info.get("behavior", {})
        merged_data = all_behavior_data.get(behavior, defaults)
        merged_data.update(data)

        all_behavior_data[behavior] = merged_data
        self.save_project_file({"behavior": all_behavior_data})

    def get_behavior(self, behavior: str) -> dict:
        """Get metadata specific to a requested behavior.

        Args:
            behavior: Behavior key to read.

        Returns:
            Dictionary of behavior metadata.
        """
        return self._project_info.get("behavior", {}).get(behavior, {})

    def remove_behavior(self, behavior: str) -> None:
        """remove behavior from project settings"""
        try:
            del self._project_info["behavior"][behavior]
            self.save_project_file()
        except KeyError:
            pass

    def update_version(self):
        """Update the version number in the metadata if it differs from the current version."""
        current_version = self._project_info.get("version")
        if current_version != version_str():
            self.save_project_file({"version": version_str()})
