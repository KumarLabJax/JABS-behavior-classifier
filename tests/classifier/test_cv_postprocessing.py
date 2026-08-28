"""Tests for postprocessed evaluation of cross-validation folds."""

import numpy as np
import pytest

from jabs.behavior.postprocessing import PostprocessingPipeline
from jabs.behavior.postprocessing.stages import BoutStitchingStage
from jabs.classifier import cv_postprocessing
from jabs.classifier.cv_postprocessing import (
    enabled_stage_configs,
    evaluate_group_with_postprocessing,
)
from jabs.classifier.inference import IdentityPrediction
from jabs.project.track_labels import TrackLabels

NONE = int(TrackLabels.Label.NONE)
NOT_BEHAVIOR = int(TrackLabels.Label.NOT_BEHAVIOR)
BEHAVIOR = int(TrackLabels.Label.BEHAVIOR)


class _FakeTrackLabels:
    """Stand-in for a per-identity ``TrackLabels`` returning a fixed vector."""

    def __init__(self, labels: np.ndarray) -> None:
        self._labels = labels

    def get_labels(self) -> np.ndarray:
        return self._labels.copy()


class _FakeVideoLabels:
    """Stand-in for ``VideoLabels`` keyed by ``(identity, behavior)``."""

    def __init__(self, labels_by_identity: dict[str, np.ndarray]) -> None:
        self._labels_by_identity = labels_by_identity
        self.requested: list[tuple[str, str]] = []

    def get_track_labels(self, identity: str, behavior: str) -> _FakeTrackLabels:
        self.requested.append((identity, behavior))
        return _FakeTrackLabels(self._labels_by_identity[identity])


class _FakePose:
    """Pose stand-in where every identity exists on the frames given as valid."""

    def __init__(self, num_frames: int, valid_by_identity: dict[int, np.ndarray]) -> None:
        self.num_frames = num_frames
        self.fps = 30
        self._valid_by_identity = valid_by_identity
        self.identities = sorted(valid_by_identity)

    def identity_mask(self, identity: int) -> np.ndarray:
        return self._valid_by_identity[identity].astype(np.int8)


class _FakeProject:
    """Minimal Project stand-in for the postprocessing evaluation path."""

    def __init__(
        self,
        labels_by_video: dict[str, dict[str, np.ndarray]],
        poses: dict[str, _FakePose],
    ) -> None:
        self.feature_dir = "features"
        self.cache_format = "hdf5"
        self._poses = poses
        self.video_labels = {
            video: _FakeVideoLabels(labels) for video, labels in labels_by_video.items()
        }
        self.opened_poses: list[str] = []

        project = self

        class _VideoManager:
            @staticmethod
            def video_path(video: str) -> str:
                return video

            @staticmethod
            def load_video_labels(video: str, _pose=None):
                return project.video_labels.get(video)

        self.video_manager = _VideoManager()

    def load_pose_est(self, video_path: str) -> _FakePose:
        """Return the pose stand-in for a video, recording the open."""
        self.opened_poses.append(video_path)
        return self._poses[video_path]


def _stitching_pipeline(max_stitch_gap: int = 1) -> PostprocessingPipeline:
    """Return a pipeline with only the stitching stage enabled."""
    return PostprocessingPipeline(
        [
            {
                "stage_name": BoutStitchingStage.__name__,
                "enabled": True,
                "parameters": {"max_stitch_gap": max_stitch_gap},
            }
        ]
    )


def _patch_prediction(monkeypatch, predictions_by_identity: dict[int, np.ndarray]) -> None:
    """Patch feature loading and inference to return fixed full-length predictions."""
    monkeypatch.setattr(
        cv_postprocessing,
        "IdentityFeatures",
        lambda video, identity, *_args, **_kwargs: identity,
    )

    def _fake_predict(_classifier, identity, _window_size) -> IdentityPrediction:
        predictions = predictions_by_identity[identity].astype(np.int8)
        confidence = np.where(predictions < 0, 0.0, 0.9).astype(np.float32)
        probabilities = np.zeros((len(predictions), 2), dtype=np.float32)
        return IdentityPrediction(
            probabilities=probabilities,
            predictions=predictions,
            confidence=confidence,
        )

    monkeypatch.setattr(cv_postprocessing, "predict_identity", _fake_predict)


@pytest.mark.parametrize(
    ("config", "expected_count"),
    [
        ([], 0),
        ([{"stage_name": "A", "enabled": False, "parameters": {}}], 0),
        ([{"stage_name": "A", "enabled": True, "parameters": {}}], 1),
        ([{"stage_name": "A", "parameters": {}}], 1),
    ],
    ids=["empty", "disabled", "enabled", "enabled_by_default"],
)
def test_enabled_stage_configs(config: list[dict], expected_count: int) -> None:
    """Only enabled stages count; a missing "enabled" key defaults to enabled."""
    assert len(enabled_stage_configs(config)) == expected_count


def test_postprocessing_uses_full_track_not_just_labeled_frames(monkeypatch) -> None:
    """Stitching must see the real frame gaps, not the gap-collapsed labeled rows.

    Frames 0-4 and 15-19 are labeled BEHAVIOR; frames 5-14 are unlabeled. The
    raw prediction is BEHAVIOR everywhere except a single NOT_BEHAVIOR frame at
    4 and another at 15, each flanked by BEHAVIOR bouts. Over the full track
    those are two separate 1-frame gaps and both stitch with
    ``max_stitch_gap=1``. If the pipeline instead ran on only the labeled rows
    the two frames would be adjacent, forming one 2-frame gap that would not
    stitch - so a perfect postprocessed score is what proves full-track
    semantics.
    """
    num_frames = 20
    labels = np.full(num_frames, NONE, dtype=np.int8)
    labels[0:5] = BEHAVIOR
    labels[15:20] = BEHAVIOR

    raw = np.full(num_frames, BEHAVIOR, dtype=np.int8)
    raw[4] = NOT_BEHAVIOR
    raw[15] = NOT_BEHAVIOR

    pose = _FakePose(num_frames, {0: np.ones(num_frames, dtype=bool)})
    project = _FakeProject(
        labels_by_video={"video.avi": {"0": labels}},
        poses={"video.avi": pose},
    )
    _patch_prediction(monkeypatch, {0: raw})

    evaluation = evaluate_group_with_postprocessing(
        classifier=object(),
        project=project,
        behavior="Walk",
        members=[("video.avi", 0)],
        pipeline=_stitching_pipeline(max_stitch_gap=1),
        behavior_settings={"window_size": 5},
        window_size=5,
    )

    assert evaluation is not None
    assert len(evaluation.truth) == 10
    assert evaluation.truth.tolist() == [BEHAVIOR] * 10
    # raw got frames 4 and 15 wrong
    assert evaluation.raw.tolist() == [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    # stitching recovered both, which is only possible on the full track
    assert evaluation.postprocessed.tolist() == [BEHAVIOR] * 10


def test_evaluation_excludes_frames_where_identity_is_absent(monkeypatch) -> None:
    """Labels are forced to NONE where the identity has no pose, matching CV's test rows."""
    num_frames = 6
    labels = np.array([BEHAVIOR] * num_frames, dtype=np.int8)
    valid = np.array([True, True, True, False, False, False])

    pose = _FakePose(num_frames, {0: valid})
    project = _FakeProject(
        labels_by_video={"video.avi": {"0": labels}},
        poses={"video.avi": pose},
    )
    raw = np.array([BEHAVIOR, BEHAVIOR, BEHAVIOR, -1, -1, -1], dtype=np.int8)
    _patch_prediction(monkeypatch, {0: raw})

    evaluation = evaluate_group_with_postprocessing(
        classifier=object(),
        project=project,
        behavior="Walk",
        members=[("video.avi", 0)],
        pipeline=_stitching_pipeline(),
        behavior_settings={"window_size": 5},
        window_size=5,
    )

    assert evaluation is not None
    assert len(evaluation.truth) == 3
    assert evaluation.raw.tolist() == [BEHAVIOR] * 3


def test_evaluation_spans_all_group_members_and_opens_each_pose_once(monkeypatch) -> None:
    """A group with several identities across videos is concatenated, pose files opened once."""
    labels = np.array([BEHAVIOR, NOT_BEHAVIOR], dtype=np.int8)
    valid = np.ones(2, dtype=bool)

    project = _FakeProject(
        labels_by_video={
            "a.avi": {"0": labels, "1": labels},
            "b.avi": {"0": labels},
        },
        poses={
            "a.avi": _FakePose(2, {0: valid, 1: valid}),
            "b.avi": _FakePose(2, {0: valid}),
        },
    )
    _patch_prediction(monkeypatch, {0: labels, 1: labels})

    evaluation = evaluate_group_with_postprocessing(
        classifier=object(),
        project=project,
        behavior="Walk",
        members=[("a.avi", 0), ("b.avi", 0), ("a.avi", 1)],
        pipeline=_stitching_pipeline(),
        behavior_settings={"window_size": 5},
        window_size=5,
    )

    assert evaluation is not None
    assert len(evaluation.truth) == 6
    assert project.opened_poses == ["a.avi", "b.avi"]


def test_evaluation_returns_none_when_no_labeled_frames(monkeypatch) -> None:
    """A group whose members have no labeled frames yields nothing to score."""
    num_frames = 4
    labels = np.full(num_frames, NONE, dtype=np.int8)
    project = _FakeProject(
        labels_by_video={"video.avi": {"0": labels}},
        poses={"video.avi": _FakePose(num_frames, {0: np.ones(num_frames, dtype=bool)})},
    )
    _patch_prediction(monkeypatch, {0: np.zeros(num_frames, dtype=np.int8)})

    evaluation = evaluate_group_with_postprocessing(
        classifier=object(),
        project=project,
        behavior="Walk",
        members=[("video.avi", 0)],
        pipeline=_stitching_pipeline(),
        behavior_settings={"window_size": 5},
        window_size=5,
    )

    assert evaluation is None


def test_evaluation_skips_videos_without_annotations(monkeypatch) -> None:
    """A member whose video has no annotation file is skipped, not fatal."""
    labels = np.array([BEHAVIOR, NOT_BEHAVIOR], dtype=np.int8)
    valid = np.ones(2, dtype=bool)
    project = _FakeProject(
        labels_by_video={"a.avi": {"0": labels}},
        poses={"a.avi": _FakePose(2, {0: valid}), "missing.avi": _FakePose(2, {0: valid})},
    )
    _patch_prediction(monkeypatch, {0: labels})

    evaluation = evaluate_group_with_postprocessing(
        classifier=object(),
        project=project,
        behavior="Walk",
        members=[("a.avi", 0), ("missing.avi", 0)],
        pipeline=_stitching_pipeline(),
        behavior_settings={"window_size": 5},
        window_size=5,
    )

    assert evaluation is not None
    assert len(evaluation.truth) == 2


def test_evaluation_honors_terminate_callback(monkeypatch) -> None:
    """The terminate callback is given a chance to abort before each identity."""
    labels = np.array([BEHAVIOR, NOT_BEHAVIOR], dtype=np.int8)
    project = _FakeProject(
        labels_by_video={"a.avi": {"0": labels}},
        poses={"a.avi": _FakePose(2, {0: np.ones(2, dtype=bool)})},
    )
    _patch_prediction(monkeypatch, {0: labels})

    def _terminate() -> None:
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        evaluate_group_with_postprocessing(
            classifier=object(),
            project=project,
            behavior="Walk",
            members=[("a.avi", 0)],
            pipeline=_stitching_pipeline(),
            behavior_settings={"window_size": 5},
            window_size=5,
            terminate_callback=_terminate,
        )
