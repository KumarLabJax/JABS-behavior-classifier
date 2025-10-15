from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd

import jabs.project.track_labels
from jabs.constants import COMPRESSION, COMPRESSION_OPTS_DEFAULT
from jabs.pose_estimation import PoseEstimation, PoseEstimationV6, PoseHashException

from .base_features import BaseFeatureGroup

# import feature modules
from .feature_base_class import Feature
from .landmark_features import LandmarkFeatureGroup
from .segmentation_features import SegmentationFeatureGroup
from .social_features import SocialFeatureGroup

FEATURE_VERSION = 16

_FEATURE_MODULES = [BaseFeatureGroup, SocialFeatureGroup, SegmentationFeatureGroup]

_EXTENDED_FEATURE_MODULES = [LandmarkFeatureGroup]

_BASE_FILTERS = {
    "social": SocialFeatureGroup.module_names(),
    "segmentation": SegmentationFeatureGroup.module_names(),
    "static_objects": LandmarkFeatureGroup.feature_map,
}

_WINDOW_FILTERS = {
    "window": list(Feature._window_operations.keys()),
    # note that fft_band features will contain a suffix for the band
    "fft": list(Feature._signal_operations.keys()),
}


class FeatureVersionException(Exception):
    """exception raised when the version of the features in the h5 file is not compatible with the current version of JABS"""

    pass


class DistanceScaleException(Exception):
    """exception raised when the distance scale factor in the h5 file don't match what the classifier expects"""

    pass


class IdentityFeatures:
    """per frame and window features for a single identity"

    Args:
        source_file:
          name of the source video or pose file, used for generating
          filenames for saving extracted features into the project
          directory. You can use None for this argument if directory
          is also set to None
        identity:
          identity to extract features for
        directory:
          path of the project directory. A value of None can be given
          to prevent saving to and loading from a project dir.
        pose_est:
          PoseEstimation object corresponding to this video
        force:
          force regeneration of per frame features even if the
          per frame feature .h5 file exists for this video/identity
        fps:
          frames per second. Used for converting angular velocity
          from degrees per frame to degrees per second
        op_settings:
          dict of optional settings to enable/disable when returning
          features. This will modify the contents returned by
          get_window_features, get_per_frame, and get_features
        cache_window:
          bool to indicate saving the window features in the cache
          directory
        compression_opts: int to indicate the compression level for
          saving features
    """

    _version = FEATURE_VERSION

    def __init__(
        self,
        source_file: str,
        identity: int,
        directory: str | Path | None,
        pose_est: PoseEstimation,
        force: bool = False,
        fps: int = 30,
        op_settings: dict | None = None,
        cache_window: bool = True,
        compression_opts: int = COMPRESSION_OPTS_DEFAULT,
    ) -> None:
        self._pose_version = pose_est.format_major_version
        self._num_frames = pose_est.num_frames
        self._fps = fps
        self._pose_hash = pose_est.hash
        self._identity = identity
        self._identity_mask = pose_est.identity_mask(identity)
        self._op_settings = dict(op_settings) if op_settings else None
        self._distance_scale_factor = (
            pose_est.cm_per_pixel if op_settings.get("cm_units", False) else None
        )
        self._compression_opts = compression_opts

        self._identity_feature_dir = (
            None
            if directory is None
            else (Path(directory) / Path(source_file).stem / str(self._identity))
        )
        self._cache_window = cache_window
        self._compute_social_features = pose_est.format_major_version >= 3

        self._compute_segmentation_features = (
            pose_est.format_major_version >= 6
            and cast(PoseEstimationV6, pose_est).has_segmentation
        )

        distance_scale = (
            self._distance_scale_factor if self._distance_scale_factor is not None else 1.0
        )

        self._feature_modules = {}
        for m in _FEATURE_MODULES:
            # don't include the social features if it is not supported by the pose file
            if not self._compute_social_features and m is SocialFeatureGroup:
                continue
            # don't include segmentation features if it is not supported by the pose file
            if not self._compute_segmentation_features and m is SegmentationFeatureGroup:
                continue
            self._feature_modules[m.name()] = m(pose_est, distance_scale)

        # load extended feature modules
        for m in _EXTENDED_FEATURE_MODULES:
            self._feature_modules[m.name()] = m(pose_est, distance_scale)

        # will hold an array that indicates if each frame is valid for this identity
        self._frame_valid = None

        # per frame features
        point_mask = pose_est.get_identity_point_mask(identity)
        self._keypoint_mask = {
            "point_mask": {
                f"point_mask {keypoint.name}": point_mask[:, keypoint.value]
                for keypoint in PoseEstimation.KeypointIndex
            }
        }

        # load or compute remaining per frame features
        if force or self._identity_feature_dir is None:
            self.__initialize_from_pose_estimation(pose_est)
        else:
            try:
                # try to load from a h5 file if it exists
                self.__load_from_file()
            except (
                OSError,
                FeatureVersionException,
                DistanceScaleException,
                PoseHashException,
            ):
                # otherwise compute the per frame features and save
                self.__initialize_from_pose_estimation(pose_est)

    def __initialize_from_pose_estimation(self, pose_est: PoseEstimation):
        """Initialize from a PoseEstimation object and save them in an h5 file

        Args:
            pose_est: PoseEstimation object used to initialize self

        Returns:
            None
        """
        # indicate this identity exists in this frame
        self._frame_valid = pose_est.identity_mask(self._identity)

        self._per_frame = self._keypoint_mask
        for key in self._feature_modules:
            self._per_frame.update(self._feature_modules[key].per_frame(self._identity))

        if self._identity_feature_dir is not None:
            self.__save_per_frame()

    def __load_from_file(self) -> None:
        """initialize from state previously saved in a h5 file on disk

        This method will throw an exception if this object was constructed with a value of None for directory

        Raises:
            OSError: if unable to open h5 file
            TypeError: if this object was constructed with a value of None for directory
            FeatureVersionException: if file version differs from current feature version
            AssertionError: if metadata shape doesn't match feature shape

        Returns:
            None
        """
        path = self._identity_feature_dir / "features.h5"
        self._per_frame = {}

        with h5py.File(path, "r") as features_h5:
            # if the version of the pose file is not the expected pose file,
            # then bail and it will get recomputed
            if features_h5.attrs["version"] != FEATURE_VERSION:
                raise FeatureVersionException

            # if the contents of the pose file changed since these features
            # were computed, then we will raise an exception and recompute
            if features_h5.attrs["pose_hash"] != self._pose_hash:
                raise PoseHashException

            # make sure distances are using the expected scale
            # if they don't match, we will need to recompute
            if self._distance_scale_factor != features_h5.attrs.get("distance_scale_factor", None):
                raise DistanceScaleException

            self._frame_valid = features_h5["frame_valid"][:]
            assert len(self._frame_valid) == self._num_frames

            # TODO
            # These class variables only exist here and are not provided elsewhere
            if self._compute_social_features:
                self._closest_identities = features_h5["closest_identities"][:]
                self._closest_fov_identities = features_h5["closest_fov_identities"][:]

            if "closest_corners" in features_h5:
                self._closest_corner = features_h5["closest_corners"][:]

            if "wall_distances" in features_h5:
                wall_distances = {}
                for key in features_h5["wall_distances"]:
                    wall_distances[key] = features_h5["wall_distances"][key][:]
                self._wall_distances = wall_distances

            if "avg_wall_length" in features_h5:
                self._avg_wall_length = features_h5["avg_wall_length"][...]

            if "closest_lixit" in features_h5:
                self._closest_lixit = features_h5["closest_lixit"][:]

            # Cache uses a space to distinguish module_name from feature_name
            for feature_key in features_h5["features/per_frame"]:
                module_name, feature_name = feature_key.split(" ", 1)
                cur_module = self._per_frame.get(module_name, {})
                cur_module[feature_name] = features_h5[f"features/per_frame/{feature_key}"][:]
                assert len(cur_module[feature_name]) == self._num_frames
                self._per_frame[module_name] = cur_module

    def __save_per_frame(self) -> None:
        """save per frame features to a h5 file

        This method will throw an exception if this object was constructed with a value of None for directory
        """
        self._identity_feature_dir.mkdir(mode=0o775, exist_ok=True, parents=True)

        file_path = self._identity_feature_dir / "features.h5"

        with h5py.File(file_path, "w") as features_h5:
            features_h5.attrs["num_frames"] = self._num_frames
            features_h5.attrs["identity"] = self._identity
            features_h5.attrs["version"] = self._version
            if self._distance_scale_factor is not None:
                features_h5.attrs["distance_scale_factor"] = self._distance_scale_factor
            features_h5.attrs["pose_hash"] = self._pose_hash
            features_h5.create_dataset(
                "frame_valid",
                data=self._frame_valid,
                compression=COMPRESSION,
                compression_opts=self._compression_opts,
            )

            if self._compute_social_features:
                closest_data = self._feature_modules[SocialFeatureGroup.name()].closest_identities

                features_h5.create_dataset(
                    "closest_identities",
                    data=closest_data.closest_identities,
                    compression=COMPRESSION,
                    compression_opts=self._compression_opts,
                )
                features_h5.create_dataset(
                    "closest_fov_identities",
                    data=closest_data.closest_fov_identities,
                    compression=COMPRESSION,
                    compression_opts=self._compression_opts,
                )

            if LandmarkFeatureGroup.name() in self._feature_modules:
                corner_info = self._feature_modules[LandmarkFeatureGroup.name()].get_corner_info(
                    self._identity
                )
                corner_data = corner_info.get_closest_corner(self._identity)
                wall_distances = corner_info.get_wall_distances(self._identity)
                avg_wall_length = corner_info.get_avg_wall_length(self._identity)
                if corner_data is not None:
                    features_h5.create_dataset(
                        "closest_corners",
                        data=corner_data,
                        compression=COMPRESSION,
                        compression_opts=self._compression_opts,
                    )
                    features_h5.create_dataset("avg_wall_length", data=avg_wall_length)
                    wall_dist_grp = features_h5.require_group("wall_distances")
                    for key, value in wall_distances.items():
                        wall_dist_grp.create_dataset(
                            key,
                            data=value,
                            compression=COMPRESSION,
                            compression_opts=self._compression_opts,
                        )

                lixit_info = self._feature_modules[LandmarkFeatureGroup.name()].get_lixit_info(
                    self._identity
                )
                lixit_data = lixit_info.get_closest_lixit(self._identity)
                if lixit_data is not None:
                    features_h5.create_dataset(
                        "closest_lixit",
                        data=lixit_data,
                        compression=COMPRESSION,
                        compression_opts=self._compression_opts,
                    )

            feature_group = features_h5.require_group("features")
            per_frame_group = feature_group.require_group("per_frame")

            per_frame_as_pd = self.merge_per_frame_features(self._per_frame)
            per_frame_as_pd = pd.DataFrame(per_frame_as_pd)
            for feature, data in per_frame_as_pd.items():
                per_frame_group.create_dataset(
                    feature,
                    data=data,
                    compression=COMPRESSION,
                    compression_opts=self._compression_opts,
                )

    def __save_window_features(self, features, window_size: int) -> None:
        """save window features to an h5 file

        This method will throw an exception if this object was constructed with a value of None for directory

        Args:
            features: window features returned from `get_window_features()` to save
            window_size: window size used

        Returns:
            None
        """
        path = self._identity_feature_dir / "features.h5"

        with h5py.File(path, "a") as features_h5:
            features_h5.attrs["num_frames"] = self._num_frames
            features_h5.attrs["identity"] = self._identity
            features_h5.attrs["version"] = self._version
            if self._distance_scale_factor is not None:
                features_h5.attrs["distance_scale_factor"] = self._distance_scale_factor
            features_h5.attrs["pose_hash"] = self._pose_hash

            feature_group = features_h5.require_group("features")
            window_group = feature_group.require_group(f"window_features_{window_size}")
            window_as_pd = self.merge_window_features(features)
            window_as_pd = pd.DataFrame(window_as_pd)
            for feature, data in window_as_pd.items():
                window_group.create_dataset(
                    feature,
                    data=data,
                    compression=COMPRESSION,
                    compression_opts=self._compression_opts,
                )

    def __load_window_features(self, window_size: int) -> dict:
        """load window features from an h5 file

        Args:
            window_size: window size specified as the number of frames
                on each side of current frame, in addition to the current frame, to
                include in the window (so if size=5, the total number of frames in the window is actually 11)

        Raises:
            OSError: if unable to open h5 file
            AttributeError: if h5 file exists but doesn't contain cached window features
            TypeError: if this object was constructed with a value of Nonefor directory
            FeatureVersionException: if file version differs from current feature version

        Returns:
            window feature dict
        """
        path = self._identity_feature_dir / "features.h5"

        window_features = {}
        with h5py.File(path, "r") as features_h5:
            # if the version of the feature file is not what we expect for
            # this version of JABS raise an exception and it will be
            # regenerated
            if features_h5.attrs["version"] != FEATURE_VERSION:
                raise FeatureVersionException

            # if the contents of the pose file changed since these features
            # were computed, then we will raise an exception and recompute
            if features_h5.attrs["pose_hash"] != self._pose_hash:
                raise PoseHashException

            # make sure distances are using the expected scale
            # if they don't match, we will need to recompute
            if self._distance_scale_factor != features_h5.attrs.get("distance_scale_factor", None):
                raise DistanceScaleException

            assert features_h5.attrs["num_frames"] == self._num_frames
            assert features_h5.attrs["identity"] == self._identity
            available_window_sizes = [
                int(x[len("window_features_") :])
                for x in features_h5["features"]
                if x.startswith("window_features_")
            ]
            if window_size not in available_window_sizes:
                raise AttributeError

            window_features = {}
            # Cache uses a space to distinguish module_name, window_name, and feature_name
            for feature_key in features_h5[f"features/window_features_{window_size}"]:
                module_name, window_name, feature_name = feature_key.split(" ", 2)
                cur_module = window_features.get(module_name, {})
                cur_window = cur_module.get(window_name, {})
                cur_window[feature_name] = features_h5[
                    f"features/window_features_{window_size}/{feature_key}"
                ][:]
                assert len(cur_window[feature_name]) == self._num_frames
                cur_module[window_name] = cur_window
                window_features[module_name] = cur_module

        return window_features

    def get_window_features(
        self, window_size: int, labels: np.ndarray | None = None, force: bool = False
    ) -> dict:
        """get window features for a given window size, computing if not previously computed and saved as h5 file

        Args:
            window_size: number of frames on each side of the current
                frame to include in the window
            labels: optional frame labels, if present then only features
                for labeled frames will be returned
            force: force regeneration of the window features even if the
               features are filtered by self.op_settings

        Note:
            if labels is None, this will also include values for frames where
            the identity does not exist. These get filtered out when filtering out
            unlabeled frames, since those frames are always unlabeled.
            h5 file already exists

        Returns:
            window features for given window size. the format is
            documented in the docstring for _compute_window_features
        """
        if force or self._identity_feature_dir is None:
            features = self.__compute_window_features(window_size)
            if self._identity_feature_dir is not None and self._cache_window:
                self.__save_window_features(features, window_size)

        else:
            try:
                # h5 file exists for this window size, load it
                features = self.__load_window_features(window_size)
            except (
                OSError,
                AttributeError,
                FeatureVersionException,
                DistanceScaleException,
                PoseHashException,
            ):
                # h5 file does not exist for this window size, the version
                # is not compatible, or the pose file changes.
                # compute the features and return after saving
                features = self.__compute_window_features(window_size)

                if self._identity_feature_dir is not None and self._cache_window:
                    self.__save_window_features(features, window_size)

        if labels is None:
            final_features = features

        else:
            # return only features for labeled frames where the identity exists
            final_features = {
                feature_module_name: {
                    window_module_name: {
                        feature_name: feature_vector[
                            (labels != jabs.project.track_labels.TrackLabels.Label.NONE)
                            & (self._identity_mask != 0)
                        ]
                        for feature_name, feature_vector in window_module.items()
                    }
                    for window_module_name, window_module in feature_module.items()
                }
                for feature_module_name, feature_module in features.items()
            }

        # filter features based on op_settings
        if self._op_settings is not None:
            final_features = self._filter_base_features_by_op(final_features)
            # window ops (e.g. 'window' or 'fft') are handled differently
            final_features = self._filter_window_features_by_op(final_features)

        return final_features

    def get_per_frame(self, labels: np.ndarray | None = None) -> dict:
        """get per frame features

        Args:
            labels: if present, only return features for labeled frames

        NOTE: if labels is None, this will include also values for frames where
            the identity does not exist. These get filtered out when filtering out
            unlabeled frames, since those frames are always unlabeled.

        Returns:
            returns per frame features in dictionary with this form

            {
                'pairwise_distances': {
                    'distance1': 1d numpy array,
                    'distance2': 1d numpy array,
                    ...
                },
                'angles': {...},
                'point_speeds': {...},
                ...
            }

            features are filtered by self.op_settings
        """
        if labels is None:
            features = self._per_frame

        else:
            # return only features for labeled frames where the identity exists
            features = {
                feature_module_name: {
                    feature_name: feature_vector[
                        (labels != jabs.project.track_labels.TrackLabels.Label.NONE)
                        & (self._identity_mask != 0)
                    ]
                    for feature_name, feature_vector in feature_module.items()
                }
                for feature_module_name, feature_module in self._per_frame.items()
            }

        if self._op_settings is not None:
            features = self._filter_base_features_by_op(features)

        return features

    def get_features(self, window_size: int) -> dict:
        """get features and corresponding frame indexes for classification

        omits frames where the identity is not valid, so 'frame_indexes' array
        may not be consecutive

        Args:
            window_size: window size to use

        Returns:
            dictionary with the following keys:

            'per_frame': dict with feature name as keys, numpy array as values
            'window': dict, see _compute_window_features
            'frame_indexes': 1D np array, maps elements of per frame and window
               feature arrays back to global frame indexes

            features are filtered by self.op_settings
        """
        per_frame_features = self.get_per_frame()
        window_features = self.get_window_features(window_size)

        indexes = np.arange(self._num_frames)[self._frame_valid == 1]

        return {
            "per_frame": per_frame_features,
            "window": window_features,
            "frame_indexes": indexes,
        }

    def _filter_base_features_by_op(self, features: dict) -> dict:
        """filter either per_frame or window features by the self._op_settings

        Args:
            features: dict of feature data

        Returns:
            filtered features
        """
        names_to_remove = []
        for setting_name, setting_val in self._op_settings.items():
            # skip if no filter assigned or we want to keep
            if setting_name not in _BASE_FILTERS or setting_val is True:
                continue
            # special case for static objects, which are nested in a second dict
            if isinstance(setting_val, dict):
                for sub_setting, sub_val in setting_val.items():
                    if not sub_val:
                        names_to_remove = names_to_remove + _BASE_FILTERS[setting_name].get(
                            sub_setting, []
                        )
            else:
                names_to_remove = names_to_remove + _BASE_FILTERS[setting_name]
        return {k: v for k, v in features.items() if k not in names_to_remove}

    def _filter_window_features_by_op(self, features: dict) -> dict:
        """filter window features by the self._op_settings

        Args:
            features: dict of window feature data (must be dict of dict of dict)

        Returns:
            filtered features
        """
        names_to_remove = []
        for setting_name, setting_val in self._op_settings.items():
            if setting_name not in _WINDOW_FILTERS or setting_val is True:
                continue
            names_to_remove = names_to_remove + _WINDOW_FILTERS[setting_name]

        # Since these names are in the second level, we need to filter them there
        filtered_features = {}
        for module_name, module_data in features.items():
            filtered_module_data = {}
            for k, v in module_data.items():
                # Handle the special case where fft_band is a prefix
                if k.startswith("fft_band") and "fft_band" in names_to_remove:
                    continue
                if k not in names_to_remove:
                    filtered_module_data[k] = v
            filtered_features[module_name] = filtered_module_data

        return filtered_features

    def __compute_window_features(self, window_size: int) -> dict:
        """compute all window features using a given window size

        Args:
            window_size: number of frames on each side of the current frame to include in the window
                (so, for example, if window_size = 5, then the total number of frames
                in the window is 11)

        Returns:
            dictionary of the form:
        {
            'angles' {
                'mean': {
                    'angle1': 1d numpy float32 array,
                    'angle2': 1d numpy float32 array,
                    'angle3': 1d numpy float32 array,
                }
                'std_dev': {
                    'angle1': 1d numpy float32 array,
                    'angle2': 1d numpy float32 array,
                    'angle3': 1d numpy float32 array,
                }
            },
            'pairwise_distances' {
                'mean': {...},
                'median': {...},
                'std_dev': {...},
                'max': {...},
                'min': {...}),
            },
            'point_speeds': {...},
            ...
        }
        """
        window_features = {}

        for key in self._feature_modules:
            window_features.update(
                self._feature_modules[key].window(self._identity, window_size, self._per_frame)
            )

        return window_features

    @classmethod
    def merge_per_frame_features(cls, features: dict) -> dict:
        """merge a dict of per-frame features

        Each element in the dict is a set of per-frame features computed for an individual animal.

        Args:
            features: list of per-frame feature instances

        Returns:
            dict of the form
        {
            'feature_1_name': feature_1_vector,
            'feature_2_name': feature_2_vector,
            'feature_3_name': feature_3_vector,
        }
        """
        merged_features = {}

        for feature_module_name, feature_module in features.items():
            if feature_module is None:
                print(f"Feature module: {feature_module_name} contains no features...")
                continue
            for feature_name, feature_vector in feature_module.items():
                merged_features[f"{feature_module_name} {feature_name}"] = feature_vector

        return merged_features

    @classmethod
    def merge_window_features(cls, features: dict) -> dict:
        """merge a dict of window features where each element in the dict is the set of window features computed for an individual animal

        Args:
            features: dict of window feature dicts

        Returns:
            dictionary of the form:
        {
            'mod1 feature_1_name': mod1_feature_1_vector,
            'mod2 feature_2_name': mod2_feature_2_vector,
            'mod1 feature_2_name': mod1_feature_2_vector,
            'mod2 feature_2_name': mod2_feature_2_vector,
            ...
        }
        """
        merged_features = {}

        for feature_module_name, feature_module in features.items():
            for window_name, window_group in feature_module.items():
                for feature_name, feature_vector in window_group.items():
                    merged_features[f"{feature_module_name} {window_name} {feature_name}"] = (
                        feature_vector
                    )

        return merged_features

    @classmethod
    def get_available_extended_features(
        cls,
        pose_version: int,
        static_objects: set[str],
        **kwargs,
    ) -> dict[str, list[str]]:
        """get all the extended features that can be used given a minimum pose version and list of available static objects

        Args:
            pose_version: integer pose version
            static_objects: list of static object names
            **kwargs: additional keyword arguments that might be used to
                determine if a feature is supported for a given pose
                file

        Returns:
            dictionary of supported extended features, where the keys
            are "feature group" name(s) and values are feature names that can
            be used from that group

        Todo:
         - social features probably should get moved into the 'extended'
          features, but they are still handled as a special case
        """
        return {
            feature_group.name(): feature_group.get_supported_feature_modules(
                pose_version, static_objects, **kwargs
            )
            for feature_group in _EXTENDED_FEATURE_MODULES
        }
