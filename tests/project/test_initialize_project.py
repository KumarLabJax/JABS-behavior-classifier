from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import jabs.feature_extraction
import jabs.scripts.initialize_project as initialize_project
from jabs.core.enums import CacheFormat
from jabs.feature_extraction.feature_timing import FeatureTimings, TimingStage


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
        timing_report,
        timing_top_n,
    ):
        captured.update(
            {
                "timing_report": timing_report,
                "timing_top_n": timing_top_n,
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
            "--timing-report",
            str(tmp_path / "timing.md"),
            "--timing-top-n",
            "10",
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
        "timing_report": tmp_path / "timing.md",
        "timing_top_n": 10,
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
        timing_report,
        timing_top_n,
    ):
        captured.update(
            {
                "timing_report": timing_report,
                "timing_top_n": timing_top_n,
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
        "timing_report": None,
        "timing_top_n": None,
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


def test_jabs_init_uses_the_canonical_default_window_size():
    """jabs-init must not carry its own copy of the default window size.

    Every other consumer reads ``jabs.feature_extraction.DEFAULT_WINDOW_SIZE``,
    so a second literal here would let the ``--window-size`` default (and the
    value quoted in its help text) drift away from the default new projects are
    created with.
    """
    assert initialize_project.DEFAULT_WINDOW_SIZE == jabs.feature_extraction.DEFAULT_WINDOW_SIZE


def test_generate_files_worker_requests_window_features_without_labels(monkeypatch):
    """The worker must not pass a stray positional argument as ``labels``.

    ``get_window_features`` takes ``labels`` as its second positional parameter, so
    anything passed there filters the returned features by label.
    """
    calls = []

    class _FakeIdentityFeatures:
        def __init__(self, *args, **kwargs):
            pass

        def get_window_features(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(jabs.feature_extraction, "IdentityFeatures", _FakeIdentityFeatures)
    pose_est = SimpleNamespace(
        format_major_version=5, identities=[], get_identity_convex_hulls=lambda i: None
    )
    project = SimpleNamespace(
        load_pose_est=lambda path: pose_est,
        video_manager=SimpleNamespace(video_path=lambda video: video),
        feature_dir="features",
        get_project_defaults=dict,
    )

    initialize_project.generate_files_worker(
        {
            "project": project,
            "video": "a.mp4",
            "identity": 0,
            "force": True,
            "window_sizes": [2, 5],
            "cache_format": CacheFormat.PARQUET.value,
        }
    )

    assert calls == [((2,), {"force": True}), ((5,), {"force": True})]


def test_jabs_init_click_rejects_top_n_without_report(tmp_path):
    """--timing-top-n is meaningless without --timing-report."""
    runner = CliRunner()
    result = runner.invoke(
        initialize_project.main, ["--timing-top-n", "5", str(tmp_path / "project")]
    )

    assert result.exit_code == 2
    assert "--timing-top-n requires --timing-report" in result.output


def test_jabs_init_click_rejects_invalid_top_n(tmp_path):
    """--timing-top-n must be at least one."""
    runner = CliRunner()
    result = runner.invoke(
        initialize_project.main,
        ["--timing-report", str(tmp_path / "t.md"), "--timing-top-n", "0", str(tmp_path)],
    )

    assert result.exit_code == 2


class _SerialPool:
    """Stand-in for multiprocessing.Pool that runs jobs in the calling process."""

    def imap_unordered(self, func, iterable):
        """Apply ``func`` to each job in order."""
        return map(func, iterable)


def _fake_project(identities_per_video: dict[str, list[int]]) -> SimpleNamespace:
    """Build the minimal project surface used by compute_project_features."""
    return SimpleNamespace(
        video_manager=SimpleNamespace(
            videos=list(identities_per_video), video_path=lambda video: video
        ),
        load_pose_est=lambda path: SimpleNamespace(identities=identities_per_video[path]),
        total_project_identities=sum(len(i) for i in identities_per_video.values()),
    )


@pytest.mark.parametrize("collect_timings", [True, False])
def test_compute_project_features_merges_worker_timings(monkeypatch, collect_timings):
    """Timings returned by each worker are merged into one collector."""
    jobs = []

    def fake_worker(params):
        jobs.append(params)
        if not params["collect_timings"]:
            return None
        timings = FeatureTimings()
        timings.record("base", "angles", TimingStage.PER_FRAME, 1.0)
        return timings

    monkeypatch.setattr(initialize_project, "generate_files_worker", fake_worker)
    project = _fake_project({"a.mp4": [0, 1], "b.mp4": [0]})

    result = initialize_project.compute_project_features(
        project, [5], False, _SerialPool(), collect_timings=collect_timings
    )

    assert len(jobs) == 3
    assert all(job["collect_timings"] is collect_timings for job in jobs)
    if collect_timings:
        (summary,) = result.summarize()
        assert summary.total_seconds == pytest.approx(3.0)
    else:
        assert result is None


@pytest.mark.parametrize("force", [True, False])
def test_write_timing_report(tmp_path, force):
    """The report is written as Markdown with a cache caveat only when not forced."""
    timings = FeatureTimings()
    timings.record("base", "angles", TimingStage.PER_FRAME, 2.0)
    timings.record("base", "point_speeds", TimingStage.PER_FRAME, 1.0)
    report_path = tmp_path / "timing.md"

    initialize_project.write_timing_report(report_path, timings, 1, {"Project": "p"}, force)

    report = report_path.read_text()
    assert "`angles`" in report
    assert "`point_speeds`" not in report
    assert "- **Project:** p" in report
    assert ("Run with `--force`" in report) is not force


def test_run_initialize_project_ignores_report_when_skipping_features(
    tmp_path, monkeypatch, capsys
):
    """A timing report is not requested from workers when features are skipped."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    class _StopAfterProject(Exception):
        pass

    def fake_project(*args, **kwargs):
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
            timing_report=tmp_path / "timing.md",
        )

    assert "--timing-report is ignored" in capsys.readouterr().out
    assert not (tmp_path / "timing.md").exists()
