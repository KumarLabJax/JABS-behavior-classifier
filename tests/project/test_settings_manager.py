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


def test_cv_grouping_regex_defaults_to_empty(mock_project):
    """cv_grouping_regex returns an empty string when not configured."""
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump({}, f)

    settings_manager = SettingsManager(mock_project.project_paths)
    assert settings_manager.cv_grouping_regex == ""


def test_cv_grouping_regex_reads_configured_value(mock_project):
    """cv_grouping_regex returns the value stored under project settings."""
    settings = {"settings": {"cv_grouping_regex": r"cage_(\d+)"}}
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump(settings, f)

    settings_manager = SettingsManager(mock_project.project_paths)
    assert settings_manager.cv_grouping_regex == r"cage_(\d+)"


def _write_behavior_settings(mock_project, behavior: str, behavior_settings: dict) -> None:
    """Write a project file containing settings for a single behavior."""
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump({"behavior": {behavior: behavior_settings}}, f)


def test_postprocessing_config_returns_stage_list(mock_project):
    """The stage list is returned in the order it was saved."""
    stages = [
        {"stage_name": "GapInterpolationStage", "enabled": False, "parameters": {}},
        {"stage_name": "BoutStitchingStage", "enabled": True, "parameters": {"max_stitch_gap": 3}},
    ]
    _write_behavior_settings(mock_project, "Walking", {"postprocessing": stages})

    settings_manager = SettingsManager(mock_project.project_paths)

    assert settings_manager.postprocessing_config("Walking") == stages


def test_postprocessing_config_defaults_to_empty(mock_project):
    """A behavior with no postprocessing configured yields an empty list."""
    _write_behavior_settings(mock_project, "Walking", {"window_size": 5})

    settings_manager = SettingsManager(mock_project.project_paths)

    assert settings_manager.postprocessing_config("Walking") == []
    assert settings_manager.postprocessing_config("Unknown") == []


@pytest.mark.parametrize(
    ("stored", "expected"),
    [({"evaluate_postprocessing_in_cv": True}, True), ({}, False)],
    ids=["enabled", "default"],
)
def test_evaluate_postprocessing_in_cv(mock_project, stored: dict, expected: bool):
    """The cross-validation evaluation flag reads back, defaulting to off."""
    _write_behavior_settings(mock_project, "Walking", stored)

    settings_manager = SettingsManager(mock_project.project_paths)

    assert settings_manager.evaluate_postprocessing_in_cv("Walking") is expected


def test_save_behavior_for_new_behavior_does_not_mutate_defaults(mock_project):
    """Saving settings for a not-yet-present behavior must not rewrite project defaults."""
    with mock_project.project_paths.project_file.open("w") as f:
        json.dump({"defaults": {"window_size": 5}, "behavior": {}}, f)

    settings_manager = SettingsManager(mock_project.project_paths)
    settings_manager.save_behavior(
        "NewBehavior",
        {
            "postprocessing": [{"stage_name": "BoutStitchingStage", "parameters": {}}],
            "evaluate_postprocessing_in_cv": True,
        },
    )

    # the new behavior inherits the defaults plus its own settings
    behavior_settings = settings_manager.get_behavior("NewBehavior")
    assert behavior_settings["window_size"] == 5
    assert behavior_settings["evaluate_postprocessing_in_cv"] is True

    # ...but the defaults themselves are untouched, so the next new behavior
    # does not silently inherit this one's postprocessing configuration
    assert settings_manager.project_settings["defaults"] == {"window_size": 5}
