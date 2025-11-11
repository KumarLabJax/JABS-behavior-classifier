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

        # atomically save combined info to file
        tmp = self._paths.project_file.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(self._project_info, f, indent=2, sort_keys=True)
        tmp.replace(self._paths.project_file)

    @property
    def project_info(self) -> dict:
        """Get a copy of the current project properties and settings.

        Returns:
            dict
        """
        return dict(self._project_info)

    @property
    def jabs_settings(self) -> dict:
        """Get a copy of general JABS settings from project file

        Returns:
            dict
        """
        return dict(self._project_info.get("settings", {}))

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

    def remove_video_from_project_file(self, video_name: str, sync=True) -> None:
        """Remove a video entry from the project if it exists.

        Note: does not remove any associated video or data files from disk, this only
         removes the entry from the project.json file.

        Args:
            video_name: Name of the video file to remove.
            sync: If True, save the project file after removal to sync the on-disk copy. Defaults to True.
        """
        video_files = self._project_info.get("video_files", {})
        video_files.pop(video_name, None)

        if sync:
            self.save_project_file()

    def set_project_metadata(self, metadata: dict, replace: bool = False) -> None:
        """Set or merge project and per-video metadata.

        By default, existing metadata is merged: new fields are added and existing fields
        are updated. If clear_existing=True, all existing metadata is cleared first.

        Args:
            metadata (dict): Dictionary containing new project-level metadata under the
                "project" key, and per-video metadata under the "videos" key. Example:
                {
                    "project": {...},
                    "videos": {
                        "video1": {...},
                        ...
                    }
                }
            replace (bool): If true, replace existing metadata instead of merge.
                Defaults to False.

        Raises:
            KeyError: If metadata for a video is provided for a video not present in the project.

        Note:
            See src/jabs/schema/metadata.py for metadata schema.
        """
        video_files = self._project_info.get("video_files", {})

        if replace:
            # Remove all existing metadata
            self._project_info.pop("metadata", None)
            for video_entry in video_files.values():
                video_entry.pop("metadata", None)

        # Merge or replace project-level metadata
        if "project" in metadata:
            existing_project_meta = self._project_info.get("metadata", {}) if not replace else {}
            merged_project_meta = dict(existing_project_meta)
            merged_project_meta.update(metadata["project"])
            self._project_info["metadata"] = merged_project_meta

        # Merge or replace per-video metadata
        for video_name, video_metadata in metadata.get("videos", {}).items():
            if video_name not in video_files:
                raise KeyError(f"Video '{video_name}' not found in project.")

            video_entry = video_files[video_name]
            existing_video_meta = video_entry.get("metadata", {}) if not replace else {}
            merged_video_meta = dict(existing_video_meta)
            merged_video_meta.update(video_metadata)
            video_entry["metadata"] = merged_video_meta

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

    def rename_behavior(self, old_name: str, new_name: str) -> None:
        """Rename a behavior in the project settings.

        Args:
            old_name: Current name of the behavior to rename.
            new_name: New name for the behavior.

        Raises:
            KeyError: If the old behavior name does not exist or
                the new behavior name already exists.
        """
        if old_name not in self._project_info.get("behavior", {}):
            raise KeyError(f"Behavior '{old_name}' not found in project.")

        if new_name in self._project_info.get("behavior", {}):
            raise KeyError(f"Behavior '{new_name}' already exists in project.")

        self._project_info["behavior"][new_name] = self._project_info["behavior"].pop(old_name)

        # if the old behavior was the selected behavior, update to new name
        if self._project_info.get("selected_behavior") == old_name:
            self._project_info["selected_behavior"] = new_name

        self.save_project_file()
