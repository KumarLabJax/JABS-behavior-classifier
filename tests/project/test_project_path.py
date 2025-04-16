import pytest
from pathlib import Path
from src.jabs.project.project_paths import ProjectPaths


@pytest.fixture
def base_path(tmp_path):
    """Fixture to provide a temporary base path."""
    return tmp_path / "test_project"


def test_project_paths_initialization(base_path):
    """Test that ProjectPaths initializes paths correctly."""
    project_paths = ProjectPaths(base_path)

    assert project_paths.project_dir == base_path
    assert project_paths.jabs_dir == base_path / "jabs"
    assert project_paths.annotations_dir == base_path / "jabs" / "annotations"
    assert project_paths.feature_dir == base_path / "jabs" / "features"
    assert project_paths.prediction_dir == base_path / "jabs" / "predictions"
    assert project_paths.classifier_dir == base_path / "jabs" / "classifiers"
    assert project_paths.archive_dir == base_path / "jabs" / "archive"
    assert project_paths.project_file == base_path / "jabs" / "project.json"
    assert project_paths.cache_dir == base_path / "jabs" / "cache"


def test_create_directories(base_path):
    """Test that create_directories creates all necessary directories."""
    project_paths = ProjectPaths(base_path)
    project_paths.create_directories()

    assert project_paths.jabs_dir.exists()
    assert project_paths.annotations_dir.exists()
    assert project_paths.feature_dir.exists()
    assert project_paths.prediction_dir.exists()
    assert project_paths.classifier_dir.exists()
    assert project_paths.archive_dir.exists()
    assert project_paths.cache_dir.exists()


def test_create_directories_idempotent(base_path):
    """Test that create_directories can be called multiple times without raising an exception."""
    project_paths = ProjectPaths(base_path)
    project_paths.create_directories()  # First call to create directories
    try:
        project_paths.create_directories()  # Second call should not raise an exception
    except Exception as e:
        pytest.fail(f"create_directories raised an exception on second call: {e}")
