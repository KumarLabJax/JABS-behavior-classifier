import json
from unittest.mock import MagicMock

import pytest

from jabs.project.settings_manager import SettingsManager


@pytest.fixture
def mock_project(tmp_path):
    """Fixture to create a mock project with necessary paths."""
    project_file = tmp_path / "project.json"
    project_paths = MagicMock()
    project_paths.project_file = project_file

    mock_project = MagicMock()
    mock_project.project_paths = project_paths

    return mock_project


def test_get_behavior(mock_project):
    """Test retrieving behavior settings."""
    # Create a mock settings file
    settings = {
        "behavior": {
            "Walking": {
                "window_size": 5,
                "balance_labels": True,
                "symmetric_behavior": False,
            }
        }
    }
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump(settings, f)

    # Create a new instance of SettingsManager to read the updated file
    settings_manager = SettingsManager(mock_project.project_paths)
    behavior_settings = settings_manager.get_behavior("Walking")
    assert behavior_settings == settings["behavior"]["Walking"]


def test_update_behavior(mock_project):
    """Test updating behavior settings."""
    # Create a mock settings file
    settings = {
        "behavior": {
            "Walking": {
                "window_size": 5,
                "balance_labels": True,
                "symmetric_behavior": False,
            }
        }
    }
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump(settings, f)

    # Create a new instance of SettingsManager to read the updated file
    settings_manager = SettingsManager(mock_project.project_paths)

    # Update behavior settings
    new_settings = {
        "window_size": 10,
        "balance_labels": False,
        "symmetric_behavior": True,
    }
    settings_manager.save_behavior("Walking", new_settings)

    # Verify the updated settings
    with mock_project.project_paths.project_file.open("r") as f:
        updated_settings = json.load(f)
    assert updated_settings["behavior"]["Walking"] == new_settings


def test_get_behavior_missing(mock_project):
    """Test retrieving settings for a missing behavior."""
    # Create a mock settings file with no behaviors
    settings = {"behavior": {}}
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump(settings, f)

    # Create a new instance of SettingsManager to read the updated file
    settings_manager = SettingsManager(mock_project.project_paths)
    behavior_settings = settings_manager.get_behavior("Running")
    assert behavior_settings == {}


def test_invalid_settings_file(mock_project):
    """Test handling of an invalid settings file."""
    # Create an invalid settings file
    with mock_project.project_paths.project_file.open("w") as f:
        f.write("invalid content")

    # Create a new instance of SettingsManager to read the updated file
    with pytest.raises(json.JSONDecodeError):
        SettingsManager(mock_project.project_paths)


def test_save_behavior(mock_project):
    """Test saving behavior settings."""
    # Create a mock settings file with initial data
    initial_settings = {
        "behavior": {
            "Walking": {
                "window_size": 5,
                "balance_labels": True,
                "symmetric_behavior": False,
            }
        }
    }
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump(initial_settings, f)

    # Create an instance of SettingsManager
    settings_manager = SettingsManager(mock_project.project_paths)

    # Save new behavior settings
    new_behavior_settings = {
        "window_size": 10,
        "balance_labels": False,
        "symmetric_behavior": True,
    }
    settings_manager.save_behavior("Running", new_behavior_settings)

    # Verify the updated settings in the file
    with mock_project.project_paths.project_file.open("r") as f:
        updated_settings = json.load(f)

    assert "Running" in updated_settings["behavior"]
    assert updated_settings["behavior"]["Running"] == new_behavior_settings
    assert updated_settings["behavior"]["Walking"] == initial_settings["behavior"]["Walking"]

    # Verify that the SettingsManager instance has the updated settings in memory
    behavior_settings = settings_manager.get_behavior("Running")
    assert behavior_settings == new_behavior_settings


def test_rename_behavior(mock_project):
    """Test renaming a behavior in the settings."""
    initial_settings = {
        "behavior": {
            "Walking": {
                "window_size": 5,
                "balance_labels": True,
                "symmetric_behavior": False,
            }
        }
    }
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump(initial_settings, f)

    settings_manager = SettingsManager(mock_project.project_paths)

    # Perform rename
    settings_manager.rename_behavior("Walking", "Walk")

    # Verify file contents
    with mock_project.project_paths.project_file.open("r") as f:
        updated_settings = json.load(f)

    assert "Walking" not in updated_settings["behavior"]
    assert "Walk" in updated_settings["behavior"]
    assert updated_settings["behavior"]["Walk"] == initial_settings["behavior"]["Walking"]

    # Verify in-memory settings
    behavior_settings = settings_manager.get_behavior("Walk")
    assert behavior_settings == initial_settings["behavior"]["Walking"]


def test_rename_behavior_updates_selected(mock_project):
    """Test that renaming also updates selected_behavior if it matches the old name."""
    initial_settings = {
        "behavior": {
            "Walking": {
                "window_size": 5,
                "balance_labels": True,
                "symmetric_behavior": False,
            }
        },
        "selected_behavior": "Walking",
    }
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump(initial_settings, f)

    settings_manager = SettingsManager(mock_project.project_paths)

    # Perform rename
    settings_manager.rename_behavior("Walking", "Walk")

    with mock_project.project_paths.project_file.open("r") as f:
        updated_settings = json.load(f)

    # selected_behavior should be updated
    assert updated_settings["selected_behavior"] == "Walk"

    # Old behavior removed, new one present
    assert "Walking" not in updated_settings["behavior"]
    assert "Walk" in updated_settings["behavior"]
    assert updated_settings["behavior"]["Walk"] == initial_settings["behavior"]["Walking"]


def test_is_video_excluded_default_false(mock_project):
    """A video with no metadata is not excluded by default."""
    settings_manager = SettingsManager(mock_project.project_paths)
    assert settings_manager.is_video_excluded("video1.avi") is False


def test_set_video_excluded_roundtrip_and_persists(mock_project):
    """Excluding a video persists to the project file and reloads."""
    settings_manager = SettingsManager(mock_project.project_paths)
    settings_manager.set_video_excluded("video1.avi", True)

    assert settings_manager.is_video_excluded("video1.avi") is True

    # a fresh manager reading the saved file sees the same state
    reloaded = SettingsManager(mock_project.project_paths)
    assert reloaded.is_video_excluded("video1.avi") is True


def test_set_video_excluded_creates_missing_entry(mock_project):
    """Excluding a video with no prior video_files entry creates one."""
    settings_manager = SettingsManager(mock_project.project_paths)
    settings_manager.set_video_excluded("new_video.avi", True)

    video_files = settings_manager.project_settings.get("video_files", {})
    assert video_files["new_video.avi"]["metadata"]["exclude_from_training"] is True


def test_set_video_excluded_toggle_back_to_included(mock_project):
    """Toggling exclusion off returns the video to included."""
    settings_manager = SettingsManager(mock_project.project_paths)
    settings_manager.set_video_excluded("video1.avi", True)
    settings_manager.set_video_excluded("video1.avi", False)

    assert settings_manager.is_video_excluded("video1.avi") is False
