from pathlib import Path

import numpy as np
import pytest

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import CacheFormat, ClassifierMode
from jabs.project.parallel_workers import collect_labeled_features
from jabs.project.track_labels import TrackLabels
from jabs.project.video_labels import VideoLabels


class _MockPose:
    """Simple pose-estimation test double."""

    def __init__(self, mask: np.ndarray):
        self._mask = mask
        self.identities = [0]

    def identity_mask(self, identity: int) -> np.ndarray:
        assert identity == 0
        return self._mask


def test_collect_labeled_features_multiclass_filters_and_aligns(
    monkeypatch, tmp_path: Path
) -> None:
    """Multiclass worker keeps only explicitly behavior-labeled, in-mask frames."""
    labels = VideoLabels("video.avi", 6)
    labels.get_track_labels("0", MULTICLASS_NONE_BEHAVIOR).label_behavior(1, 2)
    labels.get_track_labels("0", "Walk").label_behavior(3, 3)
    labels.get_track_labels("0", "Run").label_behavior(4, 4)

    pose = _MockPose(mask=np.array([1, 1, 1, 1, 0, 1], dtype=np.uint8))

    class _FakeIdentityFeatures:
        def __init__(self, *args, **kwargs):
            pass

        def get_per_frame_flat(self, labels_arr: np.ndarray) -> dict[str, np.ndarray]:
            included = labels_arr != TrackLabels.Label.NONE
            return {"per_feature": np.arange(included.sum(), dtype=np.float32)}

        def get_window_features(
            self,
            window_size: int,
            labels_arr: np.ndarray,
        ) -> dict[str, np.ndarray]:
            included = labels_arr != TrackLabels.Label.NONE
            return {"window_feature": np.arange(included.sum(), dtype=np.float32)}

        @staticmethod
        def merge_window_features(window_features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return window_features

    monkeypatch.setattr("jabs.project.parallel_workers.open_pose_file", lambda *_: pose)
    monkeypatch.setattr("jabs.project.parallel_workers.get_fps", lambda *_: 30)
    monkeypatch.setattr("jabs.project.parallel_workers._load_video_labels", lambda *_: labels)
    monkeypatch.setattr("jabs.project.parallel_workers.fe.IdentityFeatures", _FakeIdentityFeatures)

    result = collect_labeled_features(
        {
            "video": "video.avi",
            "video_path": tmp_path / "video.avi",
            "pose_path": tmp_path / "video_pose_est_v6.h5",
            "annotations_path": tmp_path / "video.json",
            "feature_dir": tmp_path / "features",
            "cache_dir": tmp_path / "cache",
            "behavior_settings": {"window_size": 3},
            "behavior_name": None,
            "behavior_names": ["Walk", "Run"],
            "classifier_mode": ClassifierMode.MULTICLASS.value,
            "cache_format": CacheFormat.HDF5.value,
        }
    )

    assert len(result["per_frame"]) == 1
    assert len(result["window"]) == 1
    assert result["per_frame"][0].shape[0] == 3
    assert result["window"][0].shape[0] == 3
    assert result["group_keys"] == [("video.avi", 0)]

    labels_by_behavior = result["labels_by_behavior"][0]
    assert np.array_equal(
        labels_by_behavior[MULTICLASS_NONE_BEHAVIOR],
        np.array(
            [
                TrackLabels.Label.BEHAVIOR,
                TrackLabels.Label.BEHAVIOR,
                TrackLabels.Label.NONE,
            ],
            dtype=np.int8,
        ),
    )
    assert np.array_equal(
        labels_by_behavior["Walk"],
        np.array(
            [
                TrackLabels.Label.NONE,
                TrackLabels.Label.NONE,
                TrackLabels.Label.BEHAVIOR,
            ],
            dtype=np.int8,
        ),
    )
    # Frame 4 had a Run label but identity mask was false there, so it is excluded.
    assert np.array_equal(
        labels_by_behavior["Run"],
        np.array(
            [
                TrackLabels.Label.NONE,
                TrackLabels.Label.NONE,
                TrackLabels.Label.NONE,
            ],
            dtype=np.int8,
        ),
    )


def test_collect_labeled_features_multiclass_requires_behavior_names(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Multiclass worker validates required behavior_names input."""
    labels = VideoLabels("video.avi", 2)
    labels.get_track_labels("0", MULTICLASS_NONE_BEHAVIOR).label_behavior(0, 0)
    pose = _MockPose(mask=np.array([1, 1], dtype=np.uint8))
    monkeypatch.setattr("jabs.project.parallel_workers.open_pose_file", lambda *_: pose)
    monkeypatch.setattr("jabs.project.parallel_workers.get_fps", lambda *_: 30)
    monkeypatch.setattr("jabs.project.parallel_workers._load_video_labels", lambda *_: labels)

    with pytest.raises(ValueError, match="behavior_names is required"):
        collect_labeled_features(
            {
                "video": "video.avi",
                "video_path": tmp_path / "video.avi",
                "pose_path": tmp_path / "video_pose_est_v6.h5",
                "annotations_path": tmp_path / "video.json",
                "feature_dir": tmp_path / "features",
                "cache_dir": tmp_path / "cache",
                "behavior_settings": {"window_size": 3},
                "behavior_name": None,
                "behavior_names": None,
                "classifier_mode": ClassifierMode.MULTICLASS.value,
                "cache_format": CacheFormat.HDF5.value,
            }
        )
