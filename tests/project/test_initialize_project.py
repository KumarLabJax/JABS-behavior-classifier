import pytest
from click.testing import CliRunner

import jabs.scripts.initialize_project as initialize_project
from jabs.core.enums import CacheFormat


def test_jabs_init_click_parses_all_existing_options(tmp_path, monkeypatch):
    """The Click entrypoint should preserve the legacy jabs-init flags and values."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}")
    project_dir = tmp_path / "project"

    captured = {}

    def fake_run_initialize_project(
        *,
        force,
        processes,
        window_sizes,
        force_pixel_distances,
        metadata_path,
        skip_feature_generation,
        project_dir,
        cache_format,
    ):
        captured.update(
            {
                "force": force,
                "processes": processes,
                "window_sizes": window_sizes,
                "force_pixel_distances": force_pixel_distances,
                "metadata_path": metadata_path,
                "skip_feature_generation": skip_feature_generation,
                "project_dir": project_dir,
                "cache_format": cache_format,
            }
        )

    monkeypatch.setattr(
        initialize_project,
        "run_initialize_project",
        fake_run_initialize_project,
    )

    runner = CliRunner()
    result = runner.invoke(
        initialize_project.main,
        [
            "-f",
            "-p",
            "8",
            "-w",
            "2",
            "-w",
            "5",
            "--force-pixel-distances",
            "--metadata",
            str(metadata_path),
            "--skip-feature-generation",
            "--cache-format",
            "hdf5",
            str(project_dir),
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "force": True,
        "processes": 8,
        "window_sizes": (2, 5),
        "force_pixel_distances": True,
        "metadata_path": metadata_path,
        "skip_feature_generation": True,
        "project_dir": project_dir,
        "cache_format": CacheFormat.HDF5,
    }


def test_jabs_init_click_uses_existing_defaults(tmp_path, monkeypatch):
    """Check defaults passed by Click entrypoint."""
    project_dir = tmp_path / "project"

    captured = {}

    def fake_run_initialize_project(
        *,
        force,
        processes,
        window_sizes,
        force_pixel_distances,
        metadata_path,
        skip_feature_generation,
        project_dir,
        cache_format,
    ):
        captured.update(
            {
                "force": force,
                "processes": processes,
                "window_sizes": window_sizes,
                "force_pixel_distances": force_pixel_distances,
                "metadata_path": metadata_path,
                "skip_feature_generation": skip_feature_generation,
                "project_dir": project_dir,
                "cache_format": cache_format,
            }
        )

    monkeypatch.setattr(
        initialize_project,
        "run_initialize_project",
        fake_run_initialize_project,
    )

    runner = CliRunner()
    result = runner.invoke(initialize_project.main, [str(project_dir)])

    assert result.exit_code == 0
    assert captured == {
        "force": False,
        "processes": initialize_project.DEFAULT_PROCESSES,
        "window_sizes": (),
        "force_pixel_distances": False,
        "metadata_path": None,
        "skip_feature_generation": False,
        "project_dir": project_dir,
        "cache_format": None,
    }


def test_jabs_init_click_rejects_invalid_window_size(tmp_path):
    """Window sizes smaller than one should still be rejected at parse time."""
    project_dir = tmp_path / "project"

    runner = CliRunner()
    result = runner.invoke(initialize_project.main, ["-w", "0", str(project_dir)])

    assert result.exit_code == 2


def test_jabs_init_click_rejects_invalid_process_count(tmp_path):
    """Process counts smaller than one should still be rejected at parse time."""
    project_dir = tmp_path / "project"

    runner = CliRunner()
    result = runner.invoke(initialize_project.main, ["-p", "0", str(project_dir)])

    assert result.exit_code == 2


def test_run_initialize_project_enables_video_frame_check(tmp_path, monkeypatch):
    """jabs-init builds the Project with the up-front video-frame check enabled.

    Phase 1 (KLAUS-505) made ``enable_video_check`` default to False, so this
    guards that batch initialization still validates that video and pose files
    agree on frame count.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    captured: dict = {}

    class _StopAfterProject(Exception):
        pass

    def fake_project(*args, **kwargs):
        captured.update(kwargs)
        raise _StopAfterProject

    monkeypatch.setattr("jabs.project.Project", fake_project)

    with pytest.raises(_StopAfterProject):
        initialize_project.run_initialize_project(
            force=False,
            processes=1,
            window_sizes=(),
            force_pixel_distances=False,
            metadata_path=None,
            skip_feature_generation=True,
            project_dir=project_dir,
        )

    assert captured.get("enable_video_check") is True
