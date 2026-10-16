from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import jabs.feature_extraction
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


def test_jabs_init_uses_the_canonical_default_window_size():
    """jabs-init must not carry its own copy of the default window size.

    Every other consumer reads ``jabs.feature_extraction.DEFAULT_WINDOW_SIZE``,
    so a second literal here would let the ``--window-size`` default (and the
    value quoted in its help text) drift away from the default new projects are
    created with.
    """
    assert initialize_project.DEFAULT_WINDOW_SIZE == jabs.feature_extraction.DEFAULT_WINDOW_SIZE


@pytest.mark.parametrize(
    ("num_videos", "processes", "expected"),
    [(15, 4, False), (16, 4, True), (200, 14, True), (3, 1, False), (4, 1, True), (0, 1, False)],
    ids=["just_below", "at_threshold", "large_project", "few_videos", "single_process", "empty"],
)
def test_use_video_tasks_needs_four_videos_per_process(
    num_videos: int, processes: int, expected: bool
) -> None:
    """Per-video tasks are used once there are at least 4 videos per worker."""
    assert initialize_project.use_video_tasks(num_videos, processes) is expected


class _FakePool:
    """Stand-in for a multiprocessing pool that records the jobs it is given."""

    def __init__(self, results_per_job: int) -> None:
        self.jobs: list[dict] = []
        self._results_per_job = results_per_job

    def imap_unordered(self, _worker, jobs):
        self.jobs = list(jobs)
        return iter([self._results_per_job] * len(self.jobs))


def _fake_project(num_videos: int, identities_per_video: int = 3) -> SimpleNamespace:
    """Project stub whose pose files all have the same identities.

    ``opened`` records every pose file that gets loaded and ``hulls`` every identity whose
    convex hulls are requested.
    """
    opened: list[str] = []
    hulls: list[int] = []
    pose = SimpleNamespace(
        identities=list(range(identities_per_video)), get_identity_convex_hulls=hulls.append
    )

    def load_pose_est(video_path):
        opened.append(video_path)
        return pose

    return SimpleNamespace(
        video_manager=SimpleNamespace(
            videos=[f"video_{i}.avi" for i in range(num_videos)], video_path=lambda video: video
        ),
        load_pose_est=load_pose_est,
        total_project_identities=num_videos * identities_per_video,
        opened=opened,
        hulls=hulls,
    )


def test_compute_project_features_uses_an_identity_job_for_small_projects() -> None:
    """Below the threshold each identity is its own job, and the producer lists identities."""
    project = _fake_project(num_videos=3)
    pool = _FakePool(results_per_job=1)

    initialize_project.compute_project_features(project, [5], False, pool, processes=4)

    assert [(job["video"], job["identities"]) for job in pool.jobs] == [
        (f"video_{video}.avi", [identity]) for video in range(3) for identity in range(3)
    ]
    assert project.opened == ["video_0.avi", "video_1.avi", "video_2.avi"]


def test_compute_project_features_uses_a_video_job_for_large_projects() -> None:
    """At the threshold each video is a single job and no pose file is opened to build it."""
    project = _fake_project(num_videos=8)
    pool = _FakePool(results_per_job=3)

    initialize_project.compute_project_features(project, [5], False, pool, processes=2)

    assert [(job["video"], job["identities"]) for job in pool.jobs] == [
        (f"video_{video}.avi", None) for video in range(8)
    ]
    assert project.opened == []


@pytest.mark.parametrize(
    ("identities", "expected_generated"),
    [(None, [0, 1, 2]), ([1], [1])],
    ids=["whole_video", "one_identity"],
)
def test_generate_files_worker_computes_the_requested_identities(
    monkeypatch, identities: list[int] | None, expected_generated: list[int]
) -> None:
    """The worker opens the pose file once, computes the requested identities and caches all hulls."""
    project = _fake_project(num_videos=1)
    generated: list[int] = []
    monkeypatch.setattr(
        initialize_project,
        "_generate_identity_files",
        lambda _project, _pose, _video, identity, *_args: generated.append(identity),
    )
    params = {
        "project": project,
        "video": "video_0.avi",
        "identities": identities,
        "force": False,
        "window_sizes": [5],
        "cache_format": CacheFormat.PARQUET.value,
    }

    count = initialize_project.generate_files_worker(params)

    assert count == len(expected_generated)
    assert generated == expected_generated
    assert project.hulls == [0, 1, 2]
    assert project.opened == ["video_0.avi"]
