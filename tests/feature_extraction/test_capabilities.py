"""Features are gated on pose capabilities and static objects, and read fps from the pose."""

import inspect
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation as pose_est
from jabs.core.enums import PoseCapability
from jabs.feature_extraction import IdentityFeatures
from jabs.feature_extraction.base_features.point_speeds import PointSpeeds
from jabs.feature_extraction.landmark_features import LandmarkFeatureGroup
from jabs.feature_extraction.landmark_features.lixit import (
    BearingToLixit,
    DistanceToLixit,
    MouseLixitAngle,
)
from jabs.feature_extraction.segmentation_features import SegmentationFeatureGroup
from jabs.feature_extraction.social_features import SocialFeatureGroup
from jabs.feature_extraction.social_features.closest_distances import ClosestDistances

_SAMPLE_V6 = Path(__file__).parent.parent / "data" / "sample_pose_est_v6.h5"

_SOCIAL = frozenset({PoseCapability.SOCIAL})
_NONE: frozenset[PoseCapability] = frozenset()


@pytest.mark.parametrize("feature_class", list(SocialFeatureGroup._features.values()))
def test_social_feature_unsupported_without_social(feature_class: type) -> None:
    """Every social feature requires SOCIAL."""
    assert not feature_class.is_supported(frozenset({PoseCapability.SEGMENTATION}), set())
    assert feature_class.is_supported(_SOCIAL, set())


def test_social_feature_requires_social() -> None:
    """Social features declare SOCIAL as their requirement."""
    assert ClosestDistances._requires == _SOCIAL


@pytest.mark.parametrize("feature_class", [DistanceToLixit, BearingToLixit])
def test_lixit_feature_supported_when_lixit_present(feature_class: type) -> None:
    """Lixit features need the lixit static object, not any capability."""
    assert feature_class.is_supported(_NONE, {"lixit"})
    assert not feature_class.is_supported(_NONE, {"corners"})


def test_mouse_lixit_angle_needs_three_keypoint_lixit() -> None:
    """MouseLixitAngle is supported only with a three-keypoint lixit."""
    assert MouseLixitAngle.is_supported(_NONE, {"lixit"}, lixit_keypoints=3)
    assert not MouseLixitAngle.is_supported(_NONE, {"lixit"}, lixit_keypoints=1)


def test_extended_features_from_capabilities_and_static_objects() -> None:
    """Available extended features follow the static objects present."""
    available = IdentityFeatures.get_available_extended_features(
        _NONE, {"lixit", "corners"}, lixit_keypoints=3
    )
    assert set(available[LandmarkFeatureGroup.name()]) == {
        *LandmarkFeatureGroup.feature_map["lixit"],
        *LandmarkFeatureGroup.feature_map["corners"],
    }
    assert LandmarkFeatureGroup.get_supported_feature_modules(_NONE, set()) == []


@pytest.fixture(scope="module")
def pose_v6(tmp_path_factory: pytest.TempPathFactory) -> Iterator[object]:
    """Open a copy of the sample v6 file (segmentation and several identities)."""
    path = tmp_path_factory.mktemp("caps_v6") / _SAMPLE_V6.name
    shutil.copyfile(_SAMPLE_V6, path)
    pose = pose_est.open_pose_file(path)
    yield pose
    pose.close()


def _with_capabilities(
    monkeypatch: pytest.MonkeyPatch, pose: object, capabilities: frozenset[PoseCapability]
) -> None:
    """Make ``pose`` report ``capabilities`` for the rest of the test."""
    monkeypatch.setattr(type(pose), "capabilities", property(lambda self: capabilities))


def test_segmentation_group_skipped_without_segmentation(
    monkeypatch: pytest.MonkeyPatch, pose_v6: object
) -> None:
    """A pose without SEGMENTATION gets no segmentation group, whatever else it holds."""
    _with_capabilities(monkeypatch, pose_v6, _SOCIAL)
    features = IdentityFeatures(None, 0, None, pose_v6)
    assert SegmentationFeatureGroup.name() not in features._feature_modules
    assert SocialFeatureGroup.name() in features._feature_modules


def test_social_group_skipped_without_social(
    monkeypatch: pytest.MonkeyPatch, pose_v6: object
) -> None:
    """A pose without SOCIAL gets no social group."""
    _with_capabilities(monkeypatch, pose_v6, frozenset({PoseCapability.SEGMENTATION}))
    features = IdentityFeatures(None, 0, None, pose_v6)
    assert SocialFeatureGroup.name() not in features._feature_modules
    assert SegmentationFeatureGroup.name() in features._feature_modules


def test_identity_features_has_no_fps_parameter() -> None:
    """fps comes from the pose; IdentityFeatures does not take it."""
    assert "fps" not in inspect.signature(IdentityFeatures).parameters


class _MovingPose:
    """Fake pose: every keypoint moves 1 pixel along x per frame."""

    def __init__(self, fps: float, num_frames: int = 10) -> None:
        self.fps = fps
        self.num_frames = num_frames

    def get_identity_poses(
        self, identity: int, scale: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the moving points and an all-true mask."""
        points = np.zeros((self.num_frames, 12, 2), dtype=np.float32)
        points[:, :, 0] = np.arange(self.num_frames)[:, np.newaxis]
        if scale is not None:
            points = points * scale
        return points, np.ones((self.num_frames, 12), dtype=bool)


def test_speed_scales_with_pose_fps() -> None:
    """A speed at 60 fps is twice the speed at 30 fps."""
    at_30 = PointSpeeds(_MovingPose(30), 1.0).per_frame(0)
    at_60 = PointSpeeds(_MovingPose(60), 1.0).per_frame(0)
    for name, values in at_30.items():
        np.testing.assert_allclose(values, 30.0)
        np.testing.assert_allclose(at_60[name], 2 * values)
