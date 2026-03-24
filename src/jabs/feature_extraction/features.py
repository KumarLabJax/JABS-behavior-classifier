import logging
from pathlib import Path
from typing import cast

import numpy as np

import jabs.project.track_labels
from jabs.core.exceptions import DistanceScaleException, FeatureVersionException
from jabs.core.types import FeatureCacheMetadata, PerFrameCacheData
from jabs.io.feature_cache.hdf5 import HDF5FeatureCacheReader, HDF5FeatureCacheWriter
from jabs.pose_estimation import PoseEstimation, PoseEstimationV6, PoseHashException

from .base_features import BaseFeatureGroup

# import feature modules
from .feature_base_class import Feature
from .landmark_features import LandmarkFeatureGroup
from .segmentation_features import SegmentationFeatureGroup
from .social_features import SocialFeatureGroup

logger = logging.getLogger(__name__)

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


class IdentityFeatures:
    """per frame and window features for a single identity

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

        self._reader = HDF5FeatureCacheReader(
            FEATURE_VERSION,
            self._pose_hash,
            self._distance_scale_factor,
        )
        self._writer = HDF5FeatureCacheWriter()

        # load or compute remaining per frame features
        if force or self._identity_feature_dir is None:
            self.__initialize_from_pose_estimation(pose_est)
        else:
            try:
                # try to load from a h5 file if it exists
                self.__load_from_file()
                logger.debug(
                    "Loaded per-frame features from cache for identity %d", self._identity
                )
            except (
                OSError,
                FeatureVersionException,
                DistanceScaleException,
                PoseHashException,
            ) as e:
                # otherwise compute the per frame features and save
                logger.info(
                    "Cache miss for identity %d per-frame features (%s); recomputing",
                    self._identity,
                    type(e).__name__,
                    exc_info=True,
                )
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
        """Initialize from state previously saved in a cache file on disk.

        Raises:
            OSError: If unable to open the cache file.
            FeatureVersionException: If the cached feature version differs from
                the current ``FEATURE_VERSION``.
            PoseHashException: If the pose file contents changed since the
                cache was written.
            DistanceScaleException: If the distance scale factor differs from
                the cached value.
        """
        cache_data = self._reader.read_per_frame(self._identity_feature_dir)
        self._frame_valid = cache_data.frame_valid
        assert len(self._frame_valid) == self._num_frames
        self._per_frame = self._unflatten_per_frame(cache_data.features)

    def __save_per_frame(self) -> None:
        """Save per-frame features to the cache."""
        closest_identities = None
        closest_fov_identities = None
        if self._compute_social_features:
            closest_data = self._feature_modules[SocialFeatureGroup.name()].closest_identities
            closest_identities = closest_data.closest_identities
            closest_fov_identities = closest_data.closest_fov_identities

        closest_corners = None
        wall_distances: dict = {}
        avg_wall_length = None
        closest_lixit = None
        if LandmarkFeatureGroup.name() in self._feature_modules:
            corner_info = self._feature_modules[LandmarkFeatureGroup.name()].get_corner_info(
                self._identity
            )
            closest_corners = corner_info.get_closest_corner(self._identity)
            if closest_corners is not None:
                wall_distances = corner_info.get_wall_distances(self._identity)
                avg_wall_length = float(corner_info.get_avg_wall_length(self._identity))
            lixit_info = self._feature_modules[LandmarkFeatureGroup.name()].get_lixit_info(
                self._identity
            )
            closest_lixit = lixit_info.get_closest_lixit(self._identity)

        metadata = FeatureCacheMetadata(
            feature_version=self._version,
            identity=self._identity,
            num_frames=self._num_frames,
            pose_hash=self._pose_hash,
            distance_scale_factor=self._distance_scale_factor,
            avg_wall_length=avg_wall_length,
        )
        cache_data = PerFrameCacheData(
            frame_valid=self._frame_valid,
            features=self.merge_per_frame_features(self._per_frame),
            closest_identities=closest_identities,
            closest_fov_identities=closest_fov_identities,
            closest_corners=closest_corners,
            closest_lixit=closest_lixit,
            wall_distances=wall_distances,
        )
        self._writer.write_per_frame(self._identity_feature_dir, metadata, cache_data)

    def __save_window_features(self, features: dict, window_size: int) -> None:
        """Save window features to the cache.

        Args:
            features: Window features as returned by ``__compute_window_features()``.
            window_size: Window size used to compute the features.
        """
        metadata = FeatureCacheMetadata(
            feature_version=self._version,
            identity=self._identity,
            num_frames=self._num_frames,
            pose_hash=self._pose_hash,
            distance_scale_factor=self._distance_scale_factor,
        )
        self._writer.write_window(
            self._identity_feature_dir,
            metadata,
            window_size,
            self.merge_window_features(features),
        )

    def __load_window_features(self, window_size: int) -> dict:
        """Load window features from the cache.

        Args:
            window_size: Window size to load.

        Raises:
            OSError: If unable to open the cache file.
            AttributeError: If the cache does not contain features for
                ``window_size``.
            FeatureVersionException: If the cached feature version differs.
            PoseHashException: If the pose file contents changed.
            DistanceScaleException: If the distance scale factor differs.

        Returns:
            Window feature dict in the nested format produced by
            ``__compute_window_features()``.
        """
        flat = self._reader.read_window(self._identity_feature_dir, window_size)
        return self._unflatten_window(flat)

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
                logger.debug(
                    "Loaded window-%d features from cache for identity %d",
                    window_size,
                    self._identity,
                )
            except (
                OSError,
                AttributeError,
                FeatureVersionException,
                DistanceScaleException,
                PoseHashException,
            ) as e:
                # h5 file does not exist for this window size, the version
                # is not compatible, or the pose file changes.
                # compute the features and return after saving
                logger.info(
                    "Cache miss for identity %d window-%d features (%s); recomputing",
                    self._identity,
                    window_size,
                    type(e).__name__,
                    exc_info=True,
                )
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
                logger.warning("Feature module '%s' contains no features", feature_module_name)
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

    @staticmethod
    def _unflatten_per_frame(flat: dict) -> dict:
        """Reconstruct a nested per-frame feature dict from a flat merged dict.

        Inverse of `merge_per_frame_features`. Splits
        ``"module_name feature_name"`` keys on the first space to recover the
        two-level nested structure expected by the rest of ``IdentityFeatures``.

        Args:
            flat: Flat dict as returned by the cache reader.

        Returns:
            Nested dict ``{module_name: {feature_name: array}}``.
        """
        nested: dict = {}
        for key, values in flat.items():
            module_name, feature_name = key.split(" ", 1)
            nested.setdefault(module_name, {})[feature_name] = values
        return nested

    @staticmethod
    def _unflatten_window(flat: dict) -> dict:
        """Reconstruct a nested window feature dict from a flat merged dict.

        Inverse of `merge_window_features`. Splits
        ``"module_name window_op feature_name"`` keys on the first two spaces
        to recover the three-level nested structure expected by the rest of
        ``IdentityFeatures``.

        Args:
            flat: Flat dict as returned by the cache reader.

        Returns:
            Nested dict ``{module_name: {window_op: {feature_name: array}}}``.
        """
        nested: dict = {}
        for key, values in flat.items():
            module_name, window_op, feature_name = key.split(" ", 2)
            nested.setdefault(module_name, {}).setdefault(window_op, {})[feature_name] = values
        return nested

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
