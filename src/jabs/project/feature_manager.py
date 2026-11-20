from typing import TYPE_CHECKING

import jabs.feature_extraction as feature_extraction
from jabs.core.enums import PoseCapability, ProjectDistanceUnit

from .project_paths import ProjectPaths
from .video_manager import VideoManager

if TYPE_CHECKING:
    from .parallel_workers import VideoScanResult


class FeatureManager:
    """Manages feature support and metadata for a JABS project.

    Determines which features are available for a project from the pose capabilities
    and static objects every usable video provides. Provides access to feature availability, distance
    units, and extended feature support, ensuring consistency across the project.

    Args:
        project_paths: Paths object for the project.
        videos: List of video filenames in the project.
        video_manager: Optional VideoManager instance (kept for callers; feature
            support comes from ``scan_results`` alone).
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

        self._can_use_social = PoseCapability.SOCIAL in self._capabilities
        self._can_use_segmentation = PoseCapability.SEGMENTATION in self._capabilities

        self._extended_features = self.__initialize_extended_features()

    def __initialize_pose_data(
        self,
        videos: list[str],
        scan_results: "dict[str, VideoScanResult]",
    ) -> None:
        """Initialize capability, static object, distance unit, and lixit data.

        Args:
            videos: List of video filenames to process for metadata extraction.
            scan_results: Per-video metadata from the project scan, keyed by
                video filename.
        """
        capability_sets: list[frozenset[PoseCapability]] = []
        static_object_sets = []
        lixit_keypoints = []
        distance_unit_valid = True

        # Single pass through all videos to gather all needed metadata
        for vid in videos:
            result = scan_results[vid]
            # an unusable video must not strip features from the others
            if result["error"] is not None:
                continue

            capability_sets.append(frozenset(PoseCapability(c) for c in result["capabilities"]))

            static_objs = result["static_objects"]
            if "lixit" in static_objs:
                lixit_keypoints.append(result["lixit_keypoints"])
            if distance_unit_valid and not result["has_cm_per_pixel"]:
                distance_unit_valid = False

            static_object_sets.append(set(static_objs))

        # Capabilities are those of ALL usable videos
        self._capabilities: frozenset[PoseCapability] = (
            frozenset.intersection(*capability_sets) if capability_sets else frozenset()
        )

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
        """Initialize extended features based on the capabilities and static objects.

        Returns:
            Dictionary of enabled extended features.
        """
        return feature_extraction.IdentityFeatures.get_available_extended_features(
            self._capabilities,
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
    def capabilities(self) -> frozenset[PoseCapability]:
        """Pose capabilities every usable video in the project provides.

        Returns:
            Intersection of the scanned capabilities of the usable videos; empty
            when there are none.
        """
        return self._capabilities

    @property
    def static_objects(self) -> set[str]:
        """Get the set of static objects in the project.

        This set contains all the static objects that are present in all pose files in the project.

        Returns:
            Set of static object names.
        """
        return self._static_objects
