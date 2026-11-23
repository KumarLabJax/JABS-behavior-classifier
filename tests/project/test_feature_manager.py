"""FeatureManager derives feature support from the scanned pose capabilities."""

from pathlib import Path

from jabs.core.enums import PoseCapability
from jabs.feature_extraction.landmark_features import LandmarkFeatureGroup
from jabs.project.feature_manager import FeatureManager
from jabs.project.parallel_workers import VideoScanResult
from jabs.project.project_paths import ProjectPaths

_SOCIAL = PoseCapability.SOCIAL
_SEGMENTATION = PoseCapability.SEGMENTATION


def _scan(
    video: str,
    capabilities: list[str],
    *,
    static_objects: list[str] | None = None,
    error: str | None = None,
) -> VideoScanResult:
    """Scan result for one video; pose files are never opened by FeatureManager."""
    return VideoScanResult(
        video=video,
        hdf5_frame_count=100,
        video_frame_count=None,
        identity_count=2,
        static_objects=static_objects or [],
        lixit_keypoints=3 if "lixit" in (static_objects or []) else 0,
        has_cm_per_pixel=True,
        capabilities=capabilities,
        error=error,
    )


def _manager(tmp_path: Path, scans: list[VideoScanResult]) -> FeatureManager:
    """FeatureManager over the given scan results."""
    paths = ProjectPaths(base_path=tmp_path)
    paths.create_directories(validate=False)
    return FeatureManager(
        paths, [s["video"] for s in scans], scan_results={s["video"]: s for s in scans}
    )


def test_capabilities_are_the_intersection_over_videos(tmp_path: Path) -> None:
    """A capability is the project's only when every video has it."""
    manager = _manager(
        tmp_path,
        [_scan("a.avi", ["segmentation", "social"]), _scan("b.avi", ["social"])],
    )
    assert manager.capabilities == frozenset({_SOCIAL})
    assert manager.can_use_social_features
    assert not manager.can_use_segmentation_features


def test_errored_video_does_not_restrict_capabilities(tmp_path: Path) -> None:
    """An unusable video is skipped."""
    manager = _manager(
        tmp_path,
        [_scan("a.avi", ["segmentation", "social"]), _scan("b.avi", [], error="bad file")],
    )
    assert manager.capabilities == frozenset({_SOCIAL, _SEGMENTATION})
    assert manager.can_use_segmentation_features


def test_no_usable_videos_has_no_capabilities(tmp_path: Path) -> None:
    """With no usable video the project has no capabilities."""
    manager = _manager(tmp_path, [_scan("a.avi", [], error="bad file")])
    assert manager.capabilities == frozenset()


def test_single_identity_project_has_no_social_features(tmp_path: Path) -> None:
    """Without SOCIAL in every video, social features are off."""
    manager = _manager(tmp_path, [_scan("a.avi", ["segmentation"])])
    assert not manager.can_use_social_features
    assert manager.can_use_segmentation_features


def test_extended_features_follow_static_objects_not_version(tmp_path: Path) -> None:
    """Landmark features need only the static object, whatever the capabilities."""
    manager = _manager(tmp_path, [_scan("a.avi", [], static_objects=["lixit"])])
    assert set(manager.extended_features[LandmarkFeatureGroup.name()]) == set(
        LandmarkFeatureGroup.feature_map["lixit"]
    )
