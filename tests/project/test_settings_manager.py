import pytest
import json
from src.jabs.project.settings_manager import SettingsManager


@pytest.fixture
def mock_project(tmp_path):
    """Fixture to create a mock project with necessary paths."""

    class MockProjectPaths:
        def __init__(self, base_path):
            self.project_file = base_path / "project.json"

    class MockProject:
        def __init__(self, base_path):
            self.project_paths = MockProjectPaths(base_path)

    return MockProject(tmp_path)


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
    assert (
        updated_settings["behavior"]["Walking"]
        == initial_settings["behavior"]["Walking"]
    )

    # Verify that the SettingsManager instance has the updated settings in memory
    behavior_settings = settings_manager.get_behavior("Running")
    assert behavior_settings == new_behavior_settings
