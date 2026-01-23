import jabs.feature_extraction as feature_extraction
from jabs.core.enums import ProjectDistanceUnit
from jabs.pose_estimation import (
    PoseEstimation,
    get_points_per_lixit,
    get_pose_file_major_version,
    get_pose_path,
    get_static_objects_in_file,
)

from .project_paths import ProjectPaths
from .video_manager import VideoManager


class FeatureManager:
    """Manages feature support and metadata for a JABS project.

    Determines which features are available for a project by analyzing pose file versions
    and static objects across all videos. Provides access to feature availability, distance
    units, and extended feature support, ensuring consistency across the project.

    Args:
        project_paths (ProjectPaths): Paths object for the project.
        videos (list[str]): List of video filenames in the project.
        video_manager: Optional VideoManager instance to use for cached pose path lookups.
    """

    def __init__(
        self,
        project_paths: ProjectPaths,
        videos: list[str],
        video_manager: VideoManager | None = None,
    ):
        """Initialize the FeatureManager."""
        self._lixit_keypoints = 0

        self._project_paths = project_paths
        self._video_manager = video_manager
        self.__initialize_pose_data(videos)

        # determine if this project can use social features or not
        # social data is available for V3+
        self._can_use_social = self._min_pose_version >= 3

        # segmentation data is available for V6+
        self._can_use_segmentation = self._min_pose_version >= 6

        self._extended_features = self.__initialize_extended_features()

    def __initialize_pose_data(self, videos: list[str]):
        """Initialize pose version, static object, distance unit, and lixit data.

        Consolidates all pose file metadata gathering into a single pass through videos
        to minimize file I/O operations.

        Args:
            videos: List of video filenames to process for metadata extraction.
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
                pose_path = get_pose_path(self._project_paths.project_dir / vid)

            # Get pose version
            pose_versions.append(get_pose_file_major_version(pose_path))

            # Get static objects
            static_objs = get_static_objects_in_file(pose_path)
            static_object_sets.append(set(static_objs))

            # Get lixit keypoints if lixit is present in this video
            if "lixit" in static_objs:
                lixit_keypoints.append(get_points_per_lixit(pose_path))

            # Check distance unit - if any video lacks cm_per_pixel, use pixels
            if distance_unit_valid:
                attrs = PoseEstimation.get_pose_file_attributes(pose_path)
                cm_per_pixel = attrs["poseest"].get("cm_per_pixel", None)
                if cm_per_pixel is None:
                    distance_unit_valid = False

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
