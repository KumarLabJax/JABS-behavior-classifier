"""Open a project, compute features, train and predict, on legacy and unified pose files.

Each project holds tiny videos beside a trimmed copy of the v4 sample pose: as
the legacy file (converted into the project cache), as its unified conversion
named ``<video>_pose.h5``, or one of each.
"""

import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

from jabs.classifier import Classifier
from jabs.classifier.inference import predict_identity
from jabs.core.enums import ClassifierType
from jabs.core.utils import hash_file
from jabs.feature_extraction import FEATURE_VERSION, IdentityFeatures
from jabs.pose_estimation.conversion_cache import ensure_converted
from jabs.project import Project, TrackLabels, VideoLabels

SAMPLE_V4 = Path(__file__).parents[1] / "data" / "sample_pose_est_v4.h5"
FRAMES = 200
BEHAVIORS = ("Walk", "Rest")
WINDOW_SIZE = 5


def _write_trimmed_v4(source: Path, destination: Path, frames: int) -> Path:
    """Copy the first ``frames`` frames of a v4 pose file, keeping every attribute."""
    with h5py.File(source, "r") as src, h5py.File(destination, "w") as dst:
        src_group = src["poseest"]
        dst_group = dst.create_group("poseest")
        dst_group.attrs.update(src_group.attrs)
        frame_count = src_group["points"].shape[0]
        for name, dataset in src_group.items():
            data = dataset[:frames] if dataset.shape[0] == frame_count else dataset[()]
            dst_group.create_dataset(name, data=data)
            dst_group[name].attrs.update(dataset.attrs)
    return destination


def _write_video(path: Path, frames: int) -> Path:
    """Write a tiny MJPG video with ``frames`` frames."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 32))
    try:
        for _ in range(frames):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.fixture(scope="module")
def pose_sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A trimmed legacy v4 file and its unified conversion, built once for the module."""
    work = tmp_path_factory.mktemp("pose-sources")
    legacy = _write_trimmed_v4(SAMPLE_V4, work / "trimmed_pose_est_v4.h5", FRAMES)
    converted = ensure_converted(legacy, hash_file(legacy), work / "cache")
    return {"legacy": legacy, "unified": converted}


def _make_project(root: Path, videos: dict[str, str], sources: dict[str, Path]) -> Path:
    """Write each video with a pose file of the given kind (``legacy`` or ``unified``)."""
    root.mkdir()
    for stem, kind in videos.items():
        _write_video(root / f"{stem}.avi", FRAMES)
        pose_name = f"{stem}_pose_est_v4.h5" if kind == "legacy" else f"{stem}_pose.h5"
        shutil.copyfile(sources[kind], root / pose_name)
    return root


def _label(project: Project) -> None:
    """Label alternating blocks of two behaviors on each video's first identity."""
    for behavior in BEHAVIORS:
        project.settings_manager.save_behavior(behavior, project.get_project_defaults())
    for video in project.video_manager.videos:
        pose = project.load_pose_est(project.video_manager.video_path(video))
        identity = pose.identities[0]
        present = np.flatnonzero(pose.identity_mask(identity))
        assert present.size >= 4 * Classifier.LABEL_THRESHOLD
        labels = VideoLabels(video, pose.num_frames)
        block = Classifier.LABEL_THRESHOLD
        for offset, behavior in enumerate(BEHAVIORS):
            track = labels.get_track_labels(str(identity), behavior)
            frames = present[: 4 * block].reshape(4, block)
            for i, chunk in enumerate(frames):
                if (i + offset) % 2 == 0:
                    track.label_behavior(chunk[0], chunk[-1])
                else:
                    track.label_not_behavior(chunk[0], chunk[-1])
        project.save_annotations(labels, pose)


def _identity_features(project: Project, video: str, identity: int) -> IdentityFeatures:
    """Return the feature accessor for one identity with the project's default settings."""
    pose = project.load_pose_est(project.video_manager.video_path(video))
    return IdentityFeatures(
        video,
        identity,
        project.feature_dir,
        pose,
        op_settings=project.get_project_defaults(),
        cache_format=project.cache_format,
    )


def _open_train_predict(project_dir: Path) -> Project:
    """Open a project, train a classifier per behavior, and predict every identity."""
    project = Project(project_dir, enable_video_check=True, enable_session_tracker=False)
    project.prepare_poses(lambda _done, _total: None)
    assert project.unusable_videos == {}
    _label(project)

    for behavior in BEHAVIORS:
        features, _ = project.get_labeled_features(behavior)
        for label in (TrackLabels.Label.BEHAVIOR, TrackLabels.Label.NOT_BEHAVIOR):
            assert (features["labels"] == label).sum() >= Classifier.LABEL_THRESHOLD
        classifier = Classifier(ClassifierType.RANDOM_FOREST)
        classifier.behavior_name = behavior
        classifier.set_project_settings(project, behavior)
        classifier.train(
            {
                "training_data": classifier.combine_data(
                    features["per_frame"], features["window"]
                ),
                "training_labels": features["labels"],
            },
            random_seed=0,
        )
        assert classifier._feature_version == FEATURE_VERSION
        project.save_classifier(classifier, behavior)

        loaded = Classifier()
        assert project.load_classifier(loaded, behavior)
        for video in project.video_manager.videos:
            pose = project.load_pose_est(project.video_manager.video_path(video))
            assert pose.identities
            for identity in pose.identities:
                prediction = predict_identity(
                    loaded, _identity_features(project, video, identity), WINDOW_SIZE
                )
                assert prediction is not None
                assert prediction.predictions.shape == (FRAMES,)
    return project


def _feature_frame(project: Project, video: str, identity: int) -> pd.DataFrame:
    """Return one identity's per-frame and window features as a single frame."""
    values = _identity_features(project, video, identity).get_features(WINDOW_SIZE)
    return Classifier.combine_data(
        pd.DataFrame(values["per_frame"]), pd.DataFrame(values["window"])
    )


def test_legacy_and_unified_projects_give_identical_features(
    tmp_path: Path, pose_sources: dict[str, Path]
) -> None:
    """A legacy-only and a unified-only project run end to end with identical features."""
    legacy = _open_train_predict(
        _make_project(tmp_path / "legacy", {"mouse": "legacy"}, pose_sources)
    )
    unified = _open_train_predict(
        _make_project(tmp_path / "unified", {"mouse": "unified"}, pose_sources)
    )

    legacy_pose = legacy.load_pose_est(legacy.video_manager.video_path("mouse.avi"))
    unified_pose = unified.load_pose_est(unified.video_manager.video_path("mouse.avi"))
    assert legacy_pose.identities == unified_pose.identities
    for identity in legacy_pose.identities:
        pd.testing.assert_frame_equal(
            _feature_frame(legacy, "mouse.avi", identity),
            _feature_frame(unified, "mouse.avi", identity),
        )


def test_mixed_project_has_the_same_feature_columns_in_every_video(
    tmp_path: Path, pose_sources: dict[str, Path]
) -> None:
    """A project mixing legacy and unified files runs end to end; columns agree."""
    project = _open_train_predict(
        _make_project(tmp_path / "mixed", {"a": "legacy", "b": "unified"}, pose_sources)
    )

    columns = {
        video: _identity_features(project, video, 0).get_per_frame_flat().keys()
        for video in project.video_manager.videos
    }
    assert list(columns["a.avi"]) == list(columns["b.avi"])
