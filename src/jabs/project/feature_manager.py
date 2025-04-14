import jabs.feature_extraction as feature_extraction
from jabs.pose_estimation import get_pose_path, get_pose_file_major_version, get_static_objects_in_file, PoseEstimation
from jabs.types import ProjectDistanceUnit
from .project_paths import ProjectPaths


class FeatureManager:
    """
    Class to manage features for a project.
    """

    def __init__(self, project_paths: ProjectPaths, videos: list[str]):
        """
        Initialize the FeatureManager.
        """

        self._project_paths = project_paths

        self.__initialize_pose_data(videos)
        self.__initialize_distance_unit(videos)

        # determine if this project can use social features or not
        # social data is available for V3+
        self._can_use_social = True if self._min_pose_version >= 3 else False

        # segmentation data is available for V6+
        self._can_use_segmentation = True if self._min_pose_version >= 6 else False

        self._extended_features = self.__initialize_extended_features()


    def __initialize_pose_data(self, videos: list[str]):
        """Initialize pose version and static object data."""
        pose_versions = []
        static_object_sets = []
        for vid in videos:
            pose_path = get_pose_path(self._project_paths.project_dir / vid)
            pose_versions.append(get_pose_file_major_version(pose_path))
            static_object_sets.append(set(get_static_objects_in_file(pose_path)))

        self._min_pose_version = min(pose_versions) if pose_versions else 0
        self._static_objects = set.intersection(*static_object_sets) if len(static_object_sets) else []

    def __initialize_distance_unit(self, videos: list[str]):
        """Determine the distance unit for the project."""
        self._distance_unit = ProjectDistanceUnit.CM
        for vid in videos:
            attrs = PoseEstimation.get_pose_file_attributes(get_pose_path(self._project_paths.project_dir / vid))
            cm_per_pixel = attrs['poseest'].get('cm_per_pixel', None)

            if cm_per_pixel is None:
                self._distance_unit = ProjectDistanceUnit.PIXEL
                break

    def __initialize_extended_features(self) -> dict:
        """
        Initialize extended features based on the pose version and static objects.
        :return: Dictionary of enabled extended features.
        """
        return feature_extraction.IdentityFeatures.get_available_extended_features(
            self._min_pose_version, self._static_objects
        )

    @property
    def can_use_social_features(self) -> bool:
        """
        Check if social features are available.
        :return: True if social features are available, False otherwise.
        """
        return self._can_use_social

    @property
    def can_use_segmentation_features(self) -> bool:
        """
        Check if segmentation features are available.
        :return: True if segmentation features are available, False otherwise.
        """
        return self._can_use_segmentation

    @property
    def extended_features(self) -> dict:
        """
        Get the enabled extended features.
        :return: Dictionary of enabled extended features.
        """
        return self._extended_features

    @property
    def is_cm_unit(self) -> bool:
        """
        Check if the distance unit is in centimeters.
        :return: True if the distance unit is in centimeters, False otherwise.
        """
        return self._distance_unit == ProjectDistanceUnit.CM

    @property
    def distance_unit(self) -> ProjectDistanceUnit:
        """
        Get the distance unit for the project.
        :return: DistanceUnit enum value representing the distance unit.
        """
        return self._distance_unit

    @property
    def min_pose_version(self) -> int:
        """
        Get the minimum pose version for the project.
        :return: Minimum pose version.
        """
        return self._min_pose_version

    @property
    def static_objects(self) -> set[str]:
        """
        Get the set of static objects in the project. This set contains all the static
        objects that are present in all pose files in the project.
        :return: Set of static objects.
        """
        return self._static_objects