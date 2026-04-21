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
