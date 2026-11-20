"""Every pose CLI accepts a legacy ``*_pose_est_v<N>.h5`` and a unified ``*_pose.h5`` file.

Each test runs once per kind on the same pose (``conftest.make_pose_file``), with
the real pose readers; only what is expensive or needs Qt (training a classifier,
rendering video) is stubbed.
"""

import sys
import types
from multiprocessing.pool import ThreadPool
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import jabs.scripts.initialize_project as initialize_project
from jabs.core.enums import PoseCapability
from jabs.project import Project
from jabs.scripts import classify
from jabs.scripts.cli import update_pose
from jabs.scripts.cli.cli import cli

from .conftest import FRAMES, POSE_KINDS, REFUSED_SOURCE, PoseFileFactory, write_video

IDENTITIES = 3


@pytest.mark.parametrize("kind", POSE_KINDS)
def test_compute_features(tmp_path: Path, make_pose_file: PoseFileFactory, kind: str) -> None:
    """compute-features writes features for every identity."""
    pose_file = make_pose_file(tmp_path, "clip", kind)
    feature_dir = tmp_path / "features"

    result = CliRunner().invoke(
        cli,
        ["compute-features", "--pose-file", str(pose_file), "--feature-dir", str(feature_dir)],
    )

    assert result.exit_code == 0, result.output
    identity_dirs = sorted(p.name for p in (feature_dir / "clip").iterdir() if p.is_dir())
    assert identity_dirs == [str(i) for i in range(IDENTITIES)]


def _fake_classifier() -> mock.Mock:
    """A binary classifier that predicts 'not behavior' everywhere."""
    classifier = mock.Mock()
    classifier.project_settings = Project.settings_by_capabilities(
        frozenset({PoseCapability.SOCIAL})
    )
    classifier.combine_data.side_effect = lambda per_frame, window: pd.concat(
        [per_frame, window], axis=1
    )
    classifier.predict_proba.side_effect = lambda data, frames: np.zeros((len(data), 2))
    classifier.derive_predictions.side_effect = lambda prob: (
        np.zeros(FRAMES, dtype=np.int8),
        np.ones(FRAMES, dtype=np.float32),
    )
    classifier.classifier_file = "fake.pkl"
    classifier.classifier_hash = "0" * 32
    return classifier


@pytest.mark.parametrize("kind", POSE_KINDS)
def test_classify(tmp_path: Path, make_pose_file: PoseFileFactory, kind: str) -> None:
    """jabs-classify writes predictions for every identity, linked to the user's file."""
    pose_file = make_pose_file(tmp_path, "clip", kind)
    out_dir = tmp_path / "out"

    classify.classify_pose(
        _fake_classifier(), pose_file, out_dir, "grooming", feature_dir=str(tmp_path / "f")
    )

    with h5py.File(out_dir / "clip_behavior.h5", "r") as h5:
        predicted = h5["predictions/grooming/predicted_class"][:]
        assert h5.attrs["pose_file"] == pose_file.name
    assert predicted.shape == (IDENTITIES, FRAMES)


@pytest.mark.parametrize("kind", POSE_KINDS)
def test_export_video_opens_the_pose(
    tmp_path: Path,
    make_pose_file: PoseFileFactory,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """export-video finds the pose beside the video and renders from it.

    The renderer needs Qt, so a stand-in records the pose it is given.
    """
    make_pose_file(tmp_path, "clip", kind)
    video = tmp_path / "clip.avi"
    video.write_bytes(b"not read")
    seen: list[tuple[int, list[int]]] = []

    def export(video_path, output_path, pose, **kwargs) -> int:
        seen.append((pose.num_frames, list(pose.identities)))
        return pose.num_frames

    fake = types.ModuleType("jabs.video_export")
    fake.VideoExportError = RuntimeError
    fake.export_overlay_video = export
    monkeypatch.setitem(sys.modules, "jabs.video_export", fake)

    result = CliRunner().invoke(cli, ["export-video", str(video)])

    assert result.exit_code == 0, result.output
    assert seen == [(FRAMES, list(range(IDENTITIES)))]


_SUBJECT = {"species": "Mus musculus", "sex": "M", "age": "P90D"}


@pytest.mark.parametrize("kind", POSE_KINDS)
def test_convert_to_nwb(tmp_path: Path, make_pose_file: PoseFileFactory, kind: str) -> None:
    """convert-to-nwb writes one file per identity and records the source format."""
    pytest.importorskip("pynwb")
    from jabs.pose_estimation import open_pose_file
    from jabs.scripts.cli.convert_to_nwb import pose_to_pose_data, run_conversion

    pose_file = make_pose_file(tmp_path, "clip", kind)
    with open_pose_file(pose_file) as pose:
        names = list(pose.external_identities)
        data = pose_to_pose_data(pose)
    assert data.metadata["pose_source_format"] == (
        "pose_est_v8" if kind == "legacy" else "jabs.pose-file"
    )
    assert "pose_format_version" not in data.metadata

    subjects = {name: {"subject_id": name, **_SUBJECT} for name in names}
    run_conversion(pose_file, tmp_path / "out.nwb", subjects=subjects)

    assert len(list(tmp_path.glob("out_*.nwb"))) == IDENTITIES


@pytest.mark.parametrize("live_kind", POSE_KINDS)
@pytest.mark.parametrize("replacement_kind", POSE_KINDS)
def test_update_pose_preflight(
    tmp_path: Path,
    make_pose_file: PoseFileFactory,
    monkeypatch: pytest.MonkeyPatch,
    live_kind: str,
    replacement_kind: str,
) -> None:
    """update-pose validates live and replacement poses of either kind."""
    project_dir = tmp_path / "project"
    new_pose_dir = tmp_path / "new_pose"
    (project_dir / "jabs").mkdir(parents=True)
    (project_dir / "jabs" / "project.json").write_text("{}")
    (project_dir / "clip.avi").touch()
    make_pose_file(project_dir, "clip", live_kind)
    replacement = make_pose_file(new_pose_dir, "clip", replacement_kind)
    monkeypatch.setattr(
        update_pose.VideoReader, "get_nframes_from_file", staticmethod(lambda _path: FRAMES)
    )

    preflight = update_pose._preflight_update_inputs(project_dir, new_pose_dir)

    assert preflight.replacement_pose_files == {"clip.avi": replacement}
    assert preflight.replacement_identity_counts == {"clip.avi": IDENTITIES}


@pytest.mark.parametrize("kind", POSE_KINDS)
def test_update_pose_reads_model_metadata(
    tmp_path: Path, make_pose_file: PoseFileFactory, kind: str
) -> None:
    """The replacement poses' model metadata is read through the pose, for either kind."""
    pose_file = make_pose_file(tmp_path, "clip", kind)

    metadata = update_pose._load_pose_file_model_metadata(pose_file)

    assert metadata is not None
    assert metadata["experiment_name"] == "synthetic_seg_and_pose"


def test_update_pose_lists_both_kinds_of_live_pose_file(
    tmp_path: Path, make_pose_file: PoseFileFactory
) -> None:
    """The live pose files replaced (and backed up) include a unified one."""
    legacy = make_pose_file(tmp_path, "clip", "legacy")
    unified = make_pose_file(tmp_path, "clip", "unified")
    make_pose_file(tmp_path, "clip2", "unified")
    (tmp_path / "._clip_pose.h5").touch()

    assert update_pose._pose_files_for_video("clip.avi", tmp_path) == sorted([legacy, unified])


def test_sample_pose_intervals_refuses_a_unified_file(
    tmp_path: Path, make_pose_file: PoseFileFactory
) -> None:
    """sample-pose-intervals copies legacy datasets, so it refuses a unified file."""
    make_pose_file(tmp_path, "clip", "unified")
    (tmp_path / "clip.avi").touch()
    batch = tmp_path / "batch.txt"
    batch.write_text("clip.avi\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = CliRunner().invoke(
        cli,
        [
            "sample-pose-intervals",
            "--batch-file",
            str(batch),
            "--root-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--out-frame-count",
            "10",
            "--only-pose",
        ],
    )

    assert "sample_pose_intervals supports legacy pose_est files only" in result.output
    assert list(out_dir.iterdir()) == []


def test_jabs_init_reports_a_refused_pose_file_and_computes_the_others(
    tmp_path: Path, make_pose_file: PoseFileFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused pose file does not stop jabs-init: the other videos get features.

    The refused file is named at the end and the exit status is 1.
    """
    # threads, not processes: spawning workers would dominate the test's run time
    monkeypatch.setattr(initialize_project, "Pool", ThreadPool)
    project_dir = tmp_path / "project"
    make_pose_file(project_dir, "good", "unified")
    write_video(project_dir / "good.avi")
    (project_dir / "bad_pose_est_v8.h5").write_bytes(REFUSED_SOURCE.read_bytes())
    write_video(project_dir / "bad.avi")

    # the project scan already marks the refused legacy file unusable
    scanned = Project(project_dir, enable_session_tracker=False)
    assert not scanned.is_usable("bad.avi")
    assert scanned.is_usable("good.avi")

    result = CliRunner().invoke(initialize_project.main, ["-p", "1", "-w", "2", str(project_dir)])

    assert result.exit_code == 1, result.output
    report = result.output.split("could not be opened")[-1]
    assert "bad.avi" in report
    assert "bad_pose_est_v8.h5" in report
    assert "good.avi" not in report
    feature_dir = scanned.feature_dir
    assert sorted(p.name for p in (feature_dir / "good").iterdir()) == [
        str(i) for i in range(IDENTITIES)
    ]
    assert not (feature_dir / "bad").exists()


def test_jabs_init_accepts_a_unified_pose_file(
    tmp_path: Path, make_pose_file: PoseFileFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """jabs-init validates a unified pose file (its frame count comes from the scan)."""
    # threads, not processes: spawning workers would dominate the test's run time
    monkeypatch.setattr(initialize_project, "Pool", ThreadPool)
    project_dir = tmp_path / "project"
    make_pose_file(project_dir, "good", "unified")
    write_video(project_dir / "good.avi")

    result = CliRunner().invoke(
        initialize_project.main, ["-p", "1", "--skip-feature-generation", str(project_dir)]
    )

    assert result.exit_code == 0, result.output
