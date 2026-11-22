"""Converting legacy poses ahead of use, and skipping videos whose pose is unusable."""

import json
import logging
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pytest

from jabs.core.enums import PoseCapability
from jabs.core.utils import hash_file
from jabs.io.internal.pose_file.convert import LegacyConversionError
from jabs.pose_estimation import PoseOpenError, conversion_cache
from jabs.pose_estimation.conversion_cache import converted_path
from jabs.project import Project, parallel_workers
from jabs.project import project as project_module
from jabs.project.parallel_workers import PoseConversionJobSpec, convert_pose

DATA_DIR = Path(__file__).parent.parent / "data"
LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)


def _add_video(project_dir: Path, video: str, pose_source: Path, pose_name: str) -> None:
    """Add a stub video and a copy of a pose file under the given name."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / video).touch()
    shutil.copyfile(pose_source, project_dir / pose_name)


def _open(project_dir: Path, process_pool: object | None = None) -> Project:
    """Open a project without a session tracker."""
    return Project(project_dir, process_pool=process_pool, enable_session_tracker=False)


def _cache_dir(project_dir: Path) -> Path:
    """Return the project's cache directory."""
    return project_dir / "jabs" / "cache"


@pytest.fixture
def legacy_project(tmp_path: Path) -> Path:
    """A project with two legacy videos (v2 and v3)."""
    project_dir = tmp_path / "proj"
    _add_video(project_dir, "a.avi", DATA_DIR / "sample_pose_est_v2.h5", "a_pose_est_v2.h5")
    _add_video(project_dir, "b.avi", DATA_DIR / "sample_pose_est_v3.h5", "b_pose_est_v3.h5")
    return project_dir


@pytest.fixture
def convert_spy(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Count convert_legacy calls while still converting."""
    spy = mock.Mock(wraps=conversion_cache.convert_legacy)
    monkeypatch.setattr(conversion_cache, "convert_legacy", spy)
    return spy


def _fake_binary_collect(seen: list[str]) -> object:
    """Return a binary feature worker that records its videos and collects nothing."""

    def collect(job: dict) -> dict:
        seen.append(job["video"])
        return {"per_frame": [], "window": [], "labels": [], "group_keys": []}

    return collect


def test_prepare_poses_converts_each_legacy_video_once(
    legacy_project: Path, convert_spy: mock.Mock
) -> None:
    """Every legacy video is converted once into the project cache; progress is (i, n)."""
    project = _open(legacy_project)
    progress: list[tuple[int, int]] = []
    project.prepare_poses(lambda i, n: progress.append((i, n)))

    assert progress == [(1, 2), (2, 2)]
    assert convert_spy.call_count == 2
    for pose in ("a_pose_est_v2.h5", "b_pose_est_v3.h5"):
        source = legacy_project / pose
        assert converted_path(source, hash_file(source), _cache_dir(legacy_project)).exists()

    project.prepare_poses(lambda i, n: None)
    assert convert_spy.call_count == 2


def test_prepare_poses_is_a_no_op_for_unified_files(
    tmp_path: Path, legacy_project: Path, convert_spy: mock.Mock
) -> None:
    """A project whose pose files are unified converts nothing and reports no progress."""
    project = _open(legacy_project)
    project.prepare_poses(lambda i, n: None)
    unified_dir = tmp_path / "unified"
    for video, pose in (("a", "a_pose_est_v2.h5"), ("b", "b_pose_est_v3.h5")):
        source = legacy_project / pose
        converted = converted_path(source, hash_file(source), _cache_dir(legacy_project))
        _add_video(unified_dir, f"{video}.avi", converted, f"{video}_pose.h5")
    convert_spy.reset_mock()

    progress: list[tuple[int, int]] = []
    _open(unified_dir).prepare_poses(lambda i, n: progress.append((i, n)))
    assert progress == []
    convert_spy.assert_not_called()
    assert not (_cache_dir(unified_dir) / "converted").exists()


def test_prepare_poses_skips_unusable_videos(
    tmp_path: Path, convert_spy: mock.Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video the scan already marked unusable is not converted."""
    monkeypatch.setenv("JABS_STRICT_POSE_CONVERSION", "1")  # lenient opens it
    project_dir = tmp_path / "proj"
    _add_video(project_dir, "a.avi", DATA_DIR / "sample_pose_est_v2.h5", "a_pose_est_v2.h5")
    _add_video(
        project_dir, "bad.avi", LEGACY_CORPUS / "v8-unknown-dataset.h5", "bad_pose_est_v8.h5"
    )
    project = _open(project_dir)
    assert not project.is_usable("bad.avi")

    progress: list[tuple[int, int]] = []
    project.prepare_poses(lambda i, n: progress.append((i, n)))
    assert progress == [(1, 1)]
    assert [c.args[0].name for c in convert_spy.call_args_list] == ["a_pose_est_v2.h5"]


def test_refusal_during_prepare_poses_marks_the_video_unusable(
    legacy_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A conversion refusal marks the video unusable; training then skips it with a WARNING."""
    real_convert = conversion_cache.convert_legacy

    def refuse_b(source: Path, destination: Path, **kwargs: object) -> None:
        if source.name.startswith("b_"):
            raise LegacyConversionError(f"{source}: shared tracklet")
        real_convert(source, destination, **kwargs)

    monkeypatch.setattr(conversion_cache, "convert_legacy", refuse_b)
    project = _open(legacy_project)
    assert project.is_usable("b.avi")

    progress: list[tuple[int, int]] = []
    with caplog.at_level(logging.WARNING, logger="jabs.project.project"):
        project.prepare_poses(lambda i, n: progress.append((i, n)))

    assert progress == [(1, 2), (2, 2)]
    assert project.unusable_videos == {"b.avi": "shared tracklet"}
    assert project.is_usable("a.avi")
    assert not project.is_usable("b.avi")
    assert any(r.levelno == logging.WARNING and "b.avi" in r.getMessage() for r in caplog.records)

    seen: list[str] = []
    monkeypatch.setattr(
        project_module, "collect_binary_labeled_features", _fake_binary_collect(seen)
    )
    ticks: list[None] = []
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="jabs.project.project"):
        project.get_labeled_features("behavior", progress_callable=lambda: ticks.append(None))
    assert seen == ["a.avi"]
    # the skipped video still advances the per-video progress
    assert len(ticks) == 2
    assert any(r.levelno == logging.WARNING and "b.avi" in r.getMessage() for r in caplog.records)


def test_multiclass_features_skip_unusable_videos(
    legacy_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-class feature collection skips unusable videos too."""
    project = _open(legacy_project)
    project.unusable_videos["b.avi"] = "refused"
    seen: list[str] = []

    def collect(job: dict) -> dict:
        seen.append(job["video"])
        return {"per_frame": [], "window": [], "labels_by_behavior": [], "group_keys": []}

    monkeypatch.setattr(project_module, "collect_multiclass_labeled_features", collect)
    project.get_multiclass_labeled_features(behavior_settings={"window_size": 5})
    assert seen == ["a.avi"]


def test_pose_open_error_in_a_feature_worker_marks_the_video_unusable(
    legacy_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that cannot open a pose does not abort training; the video becomes unusable."""
    project = _open(legacy_project)
    seen: list[str] = []
    collect_ok = _fake_binary_collect(seen)

    def collect(job: dict) -> dict:
        if job["video"] == "b.avi":
            raise PoseOpenError(job["pose_path"], "cannot convert: disk full")
        return collect_ok(job)

    monkeypatch.setattr(project_module, "collect_binary_labeled_features", collect)
    project.get_labeled_features("behavior")
    assert seen == ["a.avi"]
    assert project.unusable_videos == {"b.avi": "cannot convert: disk full"}


def test_unusable_videos_are_not_missing_features(legacy_project: Path) -> None:
    """An unusable video never needs feature computation."""
    project = _open(legacy_project)
    project.unusable_videos["b.avi"] = "refused"
    assert "b.avi" not in project.videos_missing_window_features(5)


def test_prepare_poses_in_a_pool(legacy_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a pool, conversions are submitted to it; a refusal still marks the video.

    A thread pool exercises the submit path without the minute a spawned
    process spends importing; the pickling a process pool needs is checked by
    the next test.
    """
    real_convert = conversion_cache.convert_legacy

    def refuse_b(source: Path, destination: Path, **kwargs: object) -> None:
        if source.name.startswith("b_"):
            raise LegacyConversionError(f"{source}: shared tracklet")
        real_convert(source, destination, **kwargs)

    monkeypatch.setattr(conversion_cache, "convert_legacy", refuse_b)
    with ThreadPoolExecutor(max_workers=2) as pool:
        submit = mock.Mock(wraps=pool.submit)
        project = _open(legacy_project, process_pool=mock.Mock(submit=submit))
        progress: list[tuple[int, int]] = []
        project.prepare_poses(lambda i, n: progress.append((i, n)))
    assert progress == [(1, 2), (2, 2)]
    assert sum(c.args[0] is convert_pose for c in submit.call_args_list) == 2
    assert project.unusable_videos == {"b.avi": "shared tracklet"}
    source = legacy_project / "a_pose_est_v2.h5"
    assert converted_path(source, hash_file(source), _cache_dir(legacy_project)).exists()


def test_conversion_job_and_outcome_pickle(legacy_project: Path) -> None:
    """The worker, its job and its outcome cross a process boundary."""
    job = PoseConversionJobSpec(
        video="a.avi",
        pose_path=legacy_project / "a_pose_est_v2.h5",
        cache_dir=_cache_dir(legacy_project),
        known_hash=None,
    )
    worker, job_copy = pickle.loads(pickle.dumps((convert_pose, job)))
    assert worker is convert_pose
    outcome = worker(job_copy)
    source_hash = hash_file(legacy_project / "a_pose_est_v2.h5")
    assert pickle.loads(pickle.dumps(outcome)) == ("a.avi", None, source_hash)


def test_derived_files_include_converted_and_legacy_caches(legacy_project: Path) -> None:
    """Prune finds the video's converted copies and dead reader caches, nothing else."""
    project = _open(legacy_project)
    project.prepare_poses(lambda i, n: None)
    cache = _cache_dir(legacy_project)
    converted = cache / "converted"
    source = legacy_project / "a_pose_est_v2.h5"
    current = converted_path(source, hash_file(source), cache)
    old_hash = converted / f"a.{'0' * 40}-0.0.1.h5"
    old_hash.touch()
    # another video whose name starts with "a."
    other_video = converted / f"a.2.{'1' * 40}-0.0.1.h5"
    other_video.touch()
    reader_cache = cache / "a_pose_est_v2_cache.h5"
    reader_cache.touch()

    derived = set(project.get_derived_file_paths("a.avi"))
    assert {current, old_hash, reader_cache} <= derived
    assert other_video not in derived
    b_source = legacy_project / "b_pose_est_v3.h5"
    assert converted_path(b_source, hash_file(b_source), cache) not in derived


def test_malformed_pose_marks_the_video_unusable(tmp_path: Path) -> None:
    """A pose file whose manifest breaks the scan makes the video unusable, not the project."""
    project_dir = tmp_path / "proj"
    _add_video(project_dir, "a.avi", DATA_DIR / "sample_pose_est_v2.h5", "a_pose_est_v2.h5")
    source = project_dir / "a_pose_est_v2.h5"
    converted = conversion_cache.ensure_converted(source, hash_file(source), tmp_path / "c")
    bad = project_dir / "bad_pose.h5"
    shutil.copyfile(converted, bad)
    (project_dir / "bad.avi").touch()
    with h5py.File(bad, "a") as h5:
        manifest = json.loads(h5["manifest"][()])
        manifest["components"].append(
            {**manifest["components"][0], "id": "jabs.static_objects.lixit", "shape": [1]}
        )
        del h5["manifest"]
        h5["manifest"] = json.dumps(manifest)
        h5.create_dataset("jabs/static_objects/lixit", data=np.zeros(1, dtype=np.float32))

    project = _open(project_dir)
    assert not project.is_usable("bad.avi")
    assert project.is_usable("a.avi")


def test_malformed_legacy_file_during_prepare_poses_does_not_abort_the_batch(
    tmp_path: Path,
) -> None:
    """A legacy file the scan accepts but the converter crashes on becomes unusable.

    Its static object holds strings, so ``convert_legacy`` raises ``ValueError``
    rather than ``LegacyConversionError``; the batch still converts the other video.
    """
    project_dir = tmp_path / "proj"
    _add_video(project_dir, "bad.avi", LEGACY_CORPUS / "v8-with-centers.h5", "bad_pose_est_v8.h5")
    _add_video(project_dir, "good.avi", DATA_DIR / "sample_pose_est_v2.h5", "good_pose_est_v2.h5")
    with h5py.File(project_dir / "bad_pose_est_v8.h5", "a") as h5:
        del h5["static_objects/corners"]
        h5["static_objects/corners"] = np.array([b"a", b"b"])

    project = _open(project_dir)
    assert project.is_usable("bad.avi")

    progress: list[tuple[int, int]] = []
    project.prepare_poses(lambda i, n: progress.append((i, n)))

    assert progress[-1] == (2, 2)
    assert not project.is_usable("bad.avi")
    assert "could not convert string to float" in project.unusable_videos["bad.avi"]
    assert project.is_usable("good.avi")
    good = project_dir / "good_pose_est_v2.h5"
    assert converted_path(good, hash_file(good), _cache_dir(project_dir)).exists()


def test_refusal_during_prepare_poses_refreshes_identities_and_capabilities(
    legacy_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video refused after the scan stops counting identities or restricting features."""
    real_convert = conversion_cache.convert_legacy

    def refuse_a(source: Path, destination: Path, **kwargs: object) -> None:
        if source.name.startswith("a_"):
            raise LegacyConversionError(f"{source}: refused")
        real_convert(source, destination, **kwargs)

    monkeypatch.setattr(conversion_cache, "convert_legacy", refuse_a)
    project = _open(legacy_project)
    # a (v2) has a single mouse, so the project starts without social features
    assert project.video_manager.get_video_identity_count("a.avi") == 1
    b_identities = project.video_manager.get_video_identity_count("b.avi")
    assert b_identities > 1
    assert PoseCapability.SOCIAL not in project.feature_manager.capabilities
    assert project.total_project_identities == 1 + b_identities

    project.prepare_poses(lambda i, n: None)

    assert not project.is_usable("a.avi")
    assert project.video_manager.get_video_identity_count("a.avi") == 0
    assert project.total_project_identities == b_identities
    assert PoseCapability.SOCIAL in project.feature_manager.capabilities
    assert project.feature_manager.can_use_social_features


def test_prepare_poses_raising_part_way_still_refreshes_refused_videos(
    legacy_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error after a refusal still drops the refused video's identities and features."""

    def refuse_a_then_fail(job: PoseConversionJobSpec) -> tuple[str, str | None, None]:
        if job["video"] == "a.avi":
            return job["video"], "refused", None
        raise RuntimeError("worker died")

    monkeypatch.setattr(project_module, "convert_pose", refuse_a_then_fail)
    project = _open(legacy_project)
    b_identities = project.video_manager.get_video_identity_count("b.avi")
    assert PoseCapability.SOCIAL not in project.feature_manager.capabilities

    with pytest.raises(RuntimeError, match="worker died"):
        project.prepare_poses(lambda i, n: None)

    assert not project.is_usable("a.avi")
    assert project.video_manager.get_video_identity_count("a.avi") == 0
    assert project.total_project_identities == b_identities
    assert PoseCapability.SOCIAL in project.feature_manager.capabilities


@pytest.fixture
def hash_spy(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Record every file hashed by conversion, wherever it is hashed."""
    spy = mock.Mock(wraps=hash_file)
    monkeypatch.setattr(parallel_workers, "hash_file", spy)
    monkeypatch.setattr(conversion_cache, "hash_file", spy)
    return spy


def _hashed(spy: mock.Mock) -> set[str]:
    """Return the names of the files a hash spy saw."""
    return {Path(c.args[0]).name for c in spy.call_args_list}


def test_prepare_poses_skips_hashing_unchanged_converted_files(
    legacy_project: Path, hash_spy: mock.Mock
) -> None:
    """A second open with nothing changed hashes no legacy file."""
    _open(legacy_project).prepare_poses(lambda i, n: None)
    project = _open(legacy_project)
    hash_spy.reset_mock()

    project.prepare_poses(lambda i, n: None)

    assert hash_spy.call_count == 0


def test_prepare_poses_rehashes_a_touched_file_only(
    legacy_project: Path, hash_spy: mock.Mock
) -> None:
    """A changed stat token re-hashes that file; an added video does not re-hash the others."""
    _open(legacy_project).prepare_poses(lambda i, n: None)
    a_pose = legacy_project / "a_pose_est_v2.h5"
    st = a_pose.stat()
    os.utime(a_pose, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _add_video(legacy_project, "c.avi", DATA_DIR / "sample_pose_est_v4.h5", "c_pose_est_v4.h5")
    project = _open(legacy_project)
    hash_spy.reset_mock()

    project.prepare_poses(lambda i, n: None)

    assert "b_pose_est_v3.h5" not in _hashed(hash_spy)
    assert {"a_pose_est_v2.h5", "c_pose_est_v4.h5"} <= _hashed(hash_spy)

    hash_spy.reset_mock()
    _open(legacy_project).prepare_poses(lambda i, n: None)
    assert hash_spy.call_count == 0


def test_prepare_poses_rehashes_when_the_conversion_is_missing(
    legacy_project: Path, hash_spy: mock.Mock
) -> None:
    """A recorded hash is trusted only while its conversion exists."""
    _open(legacy_project).prepare_poses(lambda i, n: None)
    shutil.rmtree(_cache_dir(legacy_project) / "converted")
    project = _open(legacy_project)
    hash_spy.reset_mock()

    project.prepare_poses(lambda i, n: None)

    assert {"a_pose_est_v2.h5", "b_pose_est_v3.h5"} <= _hashed(hash_spy)
    assert all(project.is_usable(v) for v in ("a.avi", "b.avi"))


def test_prepare_poses_reconverts_after_a_revision_bump(
    legacy_project: Path, monkeypatch: pytest.MonkeyPatch, convert_spy: mock.Mock
) -> None:
    """A recorded hash does not stop a new converter revision from converting again."""
    _open(legacy_project).prepare_poses(lambda i, n: None)
    assert convert_spy.call_count == 2
    bumped = conversion_cache._converter_revision() + 1
    monkeypatch.setattr(conversion_cache, "_converter_revision", lambda: bumped)

    _open(legacy_project).prepare_poses(lambda i, n: None)

    assert convert_spy.call_count == 4


def test_prepare_poses_survives_a_pose_file_gone_since_the_scan(legacy_project: Path) -> None:
    """A legacy file deleted after the scan marks its video unusable; the load goes on."""
    project = _open(legacy_project)
    (legacy_project / "a_pose_est_v2.h5").unlink()

    project.prepare_poses(lambda i, n: None)

    assert not project.is_usable("a.avi")
    assert project.is_usable("b.avi")


def test_convert_pose_reports_an_unsearchable_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """convert_pose returns a reason, not a raw PermissionError, when the cache cannot be searched."""
    pose_path = tmp_path / "vid_pose_est_v8.h5"
    shutil.copyfile(LEGACY_CORPUS / "v8-with-centers.h5", pose_path)
    cache = tmp_path / "cache"
    real_exists = Path.exists

    def exists(self: Path, *args: object, **kwargs: object) -> bool:
        if self.parent == cache / "converted":
            raise PermissionError(13, "Permission denied", str(self))
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)
    video, _, _ = convert_pose(
        {
            "video": "vid.avi",
            "pose_path": pose_path,
            "cache_dir": cache,
            "known_hash": hash_file(pose_path),
        }
    )
    assert video == "vid.avi"
