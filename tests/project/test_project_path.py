import pytest

from jabs.project.project_paths import ProjectPaths


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
    project_paths.create_directories(validate=False)

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
    project_paths.create_directories(validate=False)  # First call to create directories
    try:
        project_paths.create_directories(
            validate=False
        )  # Second call should not raise an exception
    except Exception as e:
        pytest.fail(f"create_directories raised an exception on second call: {e}")


def test_create_directories_validation_rejects_empty_directory(tmp_path):
    """Test that validation rejects a directory with no videos or poses."""
    project_dir = tmp_path / "empty_project"
    project_dir.mkdir()

    project_paths = ProjectPaths(project_dir)

    with pytest.raises(ValueError, match="does not appear to be a valid JABS project"):
        project_paths.create_directories(validate=True)


def test_create_directories_validation_rejects_videos_only(tmp_path):
    """Test that validation rejects a directory with only videos (no poses)."""
    project_dir = tmp_path / "videos_only"
    project_dir.mkdir()

    # Create dummy video files
    (project_dir / "video1.mp4").touch()
    (project_dir / "video2.avi").touch()

    project_paths = ProjectPaths(project_dir)

    with pytest.raises(ValueError, match="does not appear to be a valid JABS project"):
        project_paths.create_directories(validate=True)


def test_create_directories_validation_rejects_poses_only(tmp_path):
    """Test that validation rejects a directory with only poses (no videos)."""
    project_dir = tmp_path / "poses_only"
    project_dir.mkdir()

    # Create dummy pose files
    (project_dir / "video1_pose_est_v8.h5").touch()
    (project_dir / "video2_pose_est_v8.h5").touch()

    project_paths = ProjectPaths(project_dir)

    with pytest.raises(ValueError, match="does not appear to be a valid JABS project"):
        project_paths.create_directories(validate=True)


def test_create_directories_validation_accepts_videos_and_poses(tmp_path):
    """Test that validation accepts a directory with both videos and poses."""
    project_dir = tmp_path / "valid_project"
    project_dir.mkdir()

    # Create dummy video and pose files
    (project_dir / "video1.mp4").touch()
    (project_dir / "video1_pose_est_v8.h5").touch()

    project_paths = ProjectPaths(project_dir)

    # Should not raise
    project_paths.create_directories(validate=True)

    assert project_paths.jabs_dir.exists()
    assert project_paths.annotations_dir.exists()


def test_create_directories_validation_skipped_if_jabs_dir_exists(tmp_path):
    """Test that validation is skipped if jabs directory already exists."""
    project_dir = tmp_path / "existing_project"
    project_dir.mkdir()

    project_paths = ProjectPaths(project_dir)

    # Create jabs directory first (no validation needed)
    project_paths.create_directories(validate=False)

    # Now try again with validation=True but no videos/poses
    # Should succeed because jabs dir already exists
    project_paths.create_directories(validate=True)

    assert project_paths.jabs_dir.exists()
