from pathlib import Path
from typing import TYPE_CHECKING

import jabs.feature_extraction as feature_extraction
from jabs.core.enums import PoseCapability, ProjectDistanceUnit
from jabs.pose_estimation import (
    get_pose_file_major_version,
    get_pose_path,
)
from jabs.pose_estimation.discovery import is_unified_pose_file

from .project_paths import ProjectPaths
from .video_manager import VideoManager

if TYPE_CHECKING:
    from .parallel_workers import VideoScanResult

# The legacy major version a unified pose file stands in for until features are
# gated on capabilities: 6 (segmentation) when it has segmentation, else 5
# (social and static objects; static objects are still checked by name).
_UNIFIED_SEGMENTATION_VERSION = 6
_UNIFIED_VERSION = 5


def _pose_version(pose_path: Path, capabilities: list[str]) -> int:
    """Return the pose version that gates a video's features.

    A legacy file's version comes from its name; a unified file has none, so
    one is derived from its scanned capabilities.

    Args:
        pose_path: The video's pose file.
        capabilities: Its scanned ``PoseCapability`` values.

    Returns:
        The pose major version to gate features on.
    """
    if is_unified_pose_file(pose_path):
        if PoseCapability.SEGMENTATION.value in capabilities:
            return _UNIFIED_SEGMENTATION_VERSION
        return _UNIFIED_VERSION
    return get_pose_file_major_version(pose_path)


class FeatureManager:
    """Manages feature support and metadata for a JABS project.

    Determines which features are available for a project by analyzing pose file versions
    and static objects across all videos. Provides access to feature availability, distance
    units, and extended feature support, ensuring consistency across the project.

    Args:
        project_paths: Paths object for the project.
        videos: List of video filenames in the project.
        video_manager: Optional VideoManager instance used for cached pose path
            lookups. When ``None``, pose paths are resolved directly from
            ``project_paths``.
        scan_results: Per-video metadata collected by
            :func:`~jabs.project.parallel_workers.scan_video_metadata`, keyed
            by video filename. Every entry in ``videos`` must have a
            corresponding key. Required fields: ``static_objects``,
            ``lixit_keypoints``, ``has_cm_per_pixel``, ``capabilities``,
            ``error``. Videos whose scan failed (``error`` set) do not
            restrict the project's features.
    """

    def __init__(
        self,
        project_paths: ProjectPaths,
        videos: list[str],
        video_manager: VideoManager | None = None,
        *,
        scan_results: "dict[str, VideoScanResult]",
    ):
        """Initialize the FeatureManager."""
        self._lixit_keypoints = 0

        self._project_paths = project_paths
        self._video_manager = video_manager
        self.__initialize_pose_data(videos, scan_results)

        # determine if this project can use social features or not
        # social data is available for V3+
        self._can_use_social = self._min_pose_version >= 3

        # segmentation data is available for V6+
        self._can_use_segmentation = self._min_pose_version >= 6

        self._extended_features = self.__initialize_extended_features()

    def __initialize_pose_data(
        self,
        videos: list[str],
        scan_results: "dict[str, VideoScanResult]",
    ) -> None:
        """Initialize pose version, static object, distance unit, and lixit data.

        Args:
            videos: List of video filenames to process for metadata extraction.
            scan_results: Per-video metadata from the project scan, keyed by
                video filename.
        """
        pose_versions = []
        static_object_sets = []
        lixit_keypoints = []
        distance_unit_valid = True

        # Single pass through all videos to gather all needed metadata
        for vid in videos:
            # Use cached pose path if video_manager is available
            if self._video_manager:
                pose_path = self._video_manager.get_cached_pose_path(vid)
            else:
                pose_path = get_pose_path(
                    self._project_paths.video_dir / vid,
                    self._project_paths.pose_dir,
                )

            result = scan_results[vid]
            # an unusable video must not strip features from the others
            if result["error"] is not None:
                continue

            # filename regex, or scanned capabilities for a unified file - no I/O
            pose_versions.append(_pose_version(pose_path, result["capabilities"]))

            static_objs = result["static_objects"]
            if "lixit" in static_objs:
                lixit_keypoints.append(result["lixit_keypoints"])
            if distance_unit_valid and not result["has_cm_per_pixel"]:
                distance_unit_valid = False

            static_object_sets.append(set(static_objs))

        # Set pose version to minimum across all videos
        self._min_pose_version = min(pose_versions) if pose_versions else 0

        # Static objects are those present in ALL videos
        self._static_objects = (
            set.intersection(*static_object_sets) if len(static_object_sets) else []
        )

        # Determine number of keypoints used to define lixit (if present)
        # this will be used to determine if we can use single or three lixit keypoints
        # (three keypoint lixit is backwards compatible with single keypoint lixit by
        # ignoring left and right side keypoints)
        if "lixit" in self._static_objects and lixit_keypoints:
            self._lixit_keypoints = min(lixit_keypoints)

        # Set distance unit based on whether all videos have cm_per_pixel
        self._distance_unit = (
            ProjectDistanceUnit.CM if distance_unit_valid else ProjectDistanceUnit.PIXEL
        )

    def __initialize_extended_features(self) -> dict:
        """Initialize extended features based on the pose version and static objects.

        Returns:
            Dictionary of enabled extended features.
        """
        return feature_extraction.IdentityFeatures.get_available_extended_features(
            self._min_pose_version,
            self._static_objects,
            lixit_keypoints=self._lixit_keypoints,
        )

    @property
    def can_use_social_features(self) -> bool:
        """Check if social features are available.

        Returns:
            True if social features are available, False otherwise.
        """
        return self._can_use_social

    @property
    def can_use_segmentation_features(self) -> bool:
        """Check if segmentation features are available.

        Returns:
            True if segmentation features are available, False
            otherwise.
        """
        return self._can_use_segmentation

    @property
    def extended_features(self) -> dict:
        """Get the enabled extended features.

        Returns:
            Dictionary of enabled extended features.
        """
        return self._extended_features

    @property
    def is_cm_unit(self) -> bool:
        """Check if the distance unit is in centimeters.

        Returns:
            True if the distance unit is in centimeters, False
            otherwise.
        """
        return self._distance_unit == ProjectDistanceUnit.CM

    @property
    def distance_unit(self) -> ProjectDistanceUnit:
        """Get the distance unit for the project.

        Returns:
            DistanceUnit enum value representing the distance unit.
        """
        return self._distance_unit

    @property
    def min_pose_version(self) -> int:
        """Get the minimum pose version for the project.

        Returns:
            Minimum pose version.
        """
        return self._min_pose_version

    @property
    def static_objects(self) -> set[str]:
        """Get the set of static objects in the project.

        This set contains all the static objects that are present in all pose files in the project.

        Returns:
            Set of static object names.
        """
        return self._static_objects
