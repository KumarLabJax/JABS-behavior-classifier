from pathlib import Path
import typing

import h5py
import numpy as np

import src.project.track_labels
from src.pose_estimation import PoseEstimation, PoseHashException

# import feature modules
from .base_features import BaseFeatureGroup
from .social_features import SocialFeatureGroup
from .landmark_features import LandmarkFeatureGroup


FEATURE_VERSION = 4

_FEATURE_MODULES = [
    BaseFeatureGroup,
    SocialFeatureGroup
]

_EXTENDED_FEATURE_MODULES = [
    LandmarkFeatureGroup
]


class FeatureVersionException(Exception):
    pass


class DistanceScaleException(Exception):
    pass


class IdentityFeatures:
    """
    per frame and window features for a single identity
    """

    _version = FEATURE_VERSION

    def __init__(self, source_file, identity, directory, pose_est, force=False,
                 fps=30, distance_scale_factor: float = 1.0,
                 extended_features: typing.Optional[typing.Dict[str, typing.List[str]]] = None):
        """
        :param source_file: name of the source video or pose file, used for
        generating filenames for saving extracted features into the project
        directory. You can use None for this argument if directory is also set
        to None
        :param identity: identity to extract features for
        :param directory: path of the project directory. A value of None can
        be given to prevent saving to and loading from a project dir.
        :param pose_est: PoseEstimation object corresponding to this video
        :param force: force regeneration of per frame features even if the
        per frame feature .h5 file exists for this video/identity
        :param fps: frames per second. Used for converting angular velocity from
        degrees per frame to degrees per second
        :param distance_scale_factor: set to cm_per_pixel to convert pixel
        distances into cm, defaults to 1.0 (do not scale pixel coordinates)
        :param extended_features: optional extended feature configuration,
        dict with feature groups as keys, lists of feature module names as
        values. If None, all extended features are enabled.
        """

        self._num_frames = pose_est.num_frames
        self._fps = fps
        self._pose_hash = pose_est.hash
        self._identity = identity
        self._extended_features = extended_features

        # make sure distance_scale_factor is a float, passing 1 instead of 1.0
        # for using pixel units would cause some computations to use integer
        # rather than floating point
        self._distance_scale_factor = float(distance_scale_factor)

        self._identity_feature_dir = None if directory is None else (
                Path(directory) /
                Path(source_file).stem /
                str(self._identity)
        )
        self._compute_social_features = pose_est.format_major_version >= 3

        self._feature_modules = {}
        for m in _FEATURE_MODULES:
            # don't include the social features if it is not supported by
            # the pose file
            if not self._compute_social_features and m is SocialFeatureGroup:
                continue
            self._feature_modules[m.name()] = m(pose_est,
                                                self._distance_scale_factor)

        # load extended feature modules
        for m in _EXTENDED_FEATURE_MODULES:
            self._feature_modules[m.name()] = m(pose_est,
                                                self._distance_scale_factor)

        # will hold an array that indicates if each frame is valid for this
        # identity or not
        self._frame_valid = None

        # per frame features
        self._per_frame = {
            'point_mask': pose_est.get_identity_point_mask(identity)
        }

        # load or compute remaining per frame features
        if force or self._identity_feature_dir is None:
            self.__initialize_from_pose_estimation(pose_est)
        else:
            try:
                # try to load from an h5 file if it exists
                self.__load_from_file()
            except (OSError, FeatureVersionException, DistanceScaleException,
                    PoseHashException):
                # otherwise compute the per frame features and save
                self.__initialize_from_pose_estimation(pose_est)

    def __initialize_from_pose_estimation(self, pose_est):
        """
        Initialize from a PoseEstimation object and save them in an h5 file

        :param pose_est: PoseEstimation object used to initialize self
        :return: None
        """

        # indicate this identity exists in this frame
        self._frame_valid = pose_est.identity_mask(self._identity)

        for key in self._feature_modules:
            self._per_frame.update(
                self._feature_modules[key].per_frame(self._identity))

        if self._identity_feature_dir is not None:
            self.__save_per_frame()

    def __load_from_file(self):
        """
        initialize from state previously saved in a h5 file on disk
        This method will throw an exception if this object
        was constructed with a value of None for directory
        :raises OSError: if unable to open h5 file
        :raises TypeError: if this object was constructed with a value of None
        for directory
        :raises FeatureVersionException: if file version differs from current
        feature version
        :return: None
        """

        path = self._identity_feature_dir / 'per_frame.h5'

        with h5py.File(path, 'r') as features_h5:

            # if the version of the pose file is not the expected pose file,
            # then bail and it will get recomputed
            if features_h5.attrs['version'] != FEATURE_VERSION:
                raise FeatureVersionException

            # if the contents of the pose file changed since these features
            # were computed, then we will raise an exception and recompute
            if features_h5.attrs['pose_hash'] != self._pose_hash:
                raise PoseHashException

            # make sure distances are using the expected scale
            # if they don't match, we will need to recompute
            if self._distance_scale_factor != features_h5.attrs['distance_scale_factor']:
                raise DistanceScaleException

            feature_grp = features_h5['features']

            self._frame_valid = features_h5['frame_valid'][:]

            # load per frame features
            for feature in self.get_feature_names(self._compute_social_features):
                if feature in ['point_mask']:
                    continue
                self._per_frame[feature] = feature_grp[feature][:]

            assert len(self._frame_valid) == self._num_frames
            assert len(self._per_frame['pairwise_distances']) == self._num_frames
            assert len(self._per_frame['angles']) == self._num_frames

            if self._compute_social_features:
                self._closest_identities = features_h5['closest_identities'][:]
                self._closest_fov_identities = features_h5['closest_fov_identities'][:]

    def __save_per_frame(self):
        """
        save per frame features to a h5 file
        This method will throw an exception if this object
        was constructed with a value of None for directory
        """

        self._identity_feature_dir.mkdir(mode=0o775, exist_ok=True,
                                         parents=True)

        file_path = self._identity_feature_dir / 'per_frame.h5'

        with h5py.File(file_path, 'w') as features_h5:
            features_h5.attrs['num_frames'] = self._num_frames
            features_h5.attrs['identity'] = self._identity
            features_h5.attrs['version'] = self._version
            features_h5.attrs['distance_scale_factor'] = self._distance_scale_factor
            features_h5.attrs['pose_hash'] = self._pose_hash
            features_h5.create_dataset('frame_valid', data=self._frame_valid)

            grp = features_h5.create_group('features')
            for feature in self.get_feature_names(self._compute_social_features):
                # point mask is obtained from the pose file, don't save it
                if feature in ['point_mask']:
                    continue
                grp.create_dataset(feature, data=self._per_frame[feature])

            if self._compute_social_features:
                closest_data = self._feature_modules[SocialFeatureGroup.name()].closest_identities
                features_h5['closest_identities'] = closest_data.closest_identities
                features_h5['closest_fov_identities'] = closest_data.closest_fov_identities

    def __save_window_features(self, features, window_size):
        """
        save window features to an h5 file
        This method will throw an exception if this object
        was constructed with a value of None for directory
        :param features: window features to save
        :param window_size:
        :return: None
        """
        path = self._identity_feature_dir / f"window_features_{window_size}.h5"

        with h5py.File(path, 'w') as features_h5:
            features_h5.attrs['window_size'] = window_size
            features_h5.attrs['num_frames'] = self._num_frames
            features_h5.attrs['identity'] = self._identity
            features_h5.attrs['version'] = self._version
            features_h5.attrs['distance_scale_factor'] = self._distance_scale_factor
            features_h5.attrs['pose_hash'] = self._pose_hash

            grp = features_h5.create_group('features')

            for feature in features:
                for op in features[feature]:
                    grp.create_dataset(f'{feature}/{op}',
                                       data=features[feature][op])

    def __load_window_features(self, window_size):
        """
        load window features from an h5 file
        :param window_size: window size specified as the number of frames on
        each side of current frame, in addition to the current frame, to
        include in the window
        (so if size=5, the total number of frames in the window is actually 11)
        :raises OSError: if unable to open h5 file
        :raises TypeError: if this object was constructed with a value of None
        for directory
        :raises FeatureVersionException: if file version differs from current
        feature version

        :return: window feature dict
        """
        path = self._identity_feature_dir / f"window_features_{window_size}.h5"

        window_features = {}
        with h5py.File(path, 'r') as features_h5:

            # if the version of the feature file is not what we expect for
            # this version of JABS raise an exception and it will be
            # regenerated
            if features_h5.attrs['version'] != FEATURE_VERSION:
                raise FeatureVersionException

            # if the contents of the pose file changed since these features
            # were computed, then we will raise an exception and recompute
            if features_h5.attrs['pose_hash'] != self._pose_hash:
                raise PoseHashException

            # make sure distances are using the expected scale
            # if they don't match, we will need to recompute
            if self._distance_scale_factor != features_h5.attrs['distance_scale_factor']:
                raise DistanceScaleException

            size_attr = features_h5.attrs['window_size']

            assert features_h5.attrs['num_frames'] == self._num_frames
            assert features_h5.attrs['identity'] == self._identity
            assert size_attr == window_size

            feature_grp = features_h5['features']

            for feature_name in self.get_feature_names(self._compute_social_features):
                # point_mask is loaded from the post estimation, not the
                # feature file
                if feature_name == 'point_mask':
                    continue
                window_features[feature_name] = {}
                for op in feature_grp[feature_name].keys():
                    window_features[feature_name][op] = \
                        feature_grp[f'{feature_name}/{op}'][:]
                    assert len(
                        window_features[feature_name][op]) == self._num_frames

            return window_features

    def get_window_features(self, window_size: int, use_social: bool,
                            labels=None, force: bool = False):
        """
        get window features for a given window size, computing if not previously
        computed and saved as h5 file
        :param window_size: number of frames on each side of the current frame to
        include in the window
        :param use_social:
        :param labels: optional frame labels, if present then only features for
        labeled frames will be returned
        NOTE: if labels is None, this will also include values for frames where
        the identity does not exist. These get filtered out when filtering out
        unlabeled frames, since those frames are always unlabeled.
        :param force: force regeneration of the window features even if the
        h5 file already exists
        :return: window features for given window size. the format is documented
        in the docstring for _compute_window_features
        """

        if force or self._identity_feature_dir is None:
            features = self.__compute_window_features(window_size)
            if self._identity_feature_dir is not None:
                self.__save_window_features(features, window_size)

        else:
            try:
                # h5 file exists for this window size, load it
                features = self.__load_window_features(window_size)
            except (OSError, FeatureVersionException, DistanceScaleException,
                    PoseHashException):
                # h5 file does not exist for this window size, the version
                # is not compatible, or the pose file changes.
                # compute the features and return after saving
                features = self.__compute_window_features(window_size)

                if self._identity_feature_dir is not None:
                    self.__save_window_features(features, window_size)

        feature_intersection = self.get_feature_names(
            use_social, self._extended_features)
        feature_intersection &= set(features.keys())

        if labels is None:
            final_features = features

        else:
            # return only features for labeled frames
            filtered_features = {}

            for key in features:
                filtered_features[key] = {}
                for op in features[key]:
                    filtered_features[key][op] = features[key][op][labels != src.project.track_labels.TrackLabels.Label.NONE]

            final_features = filtered_features

        return {
            feature_name: final_features[feature_name]
            for feature_name in feature_intersection
        }

    def get_per_frame(self, use_social: bool, labels=None):
        """
        get per frame features
        :param use_social:
        :param labels: if present, only return features for labeled frames
        NOTE: if labels is None, this will include also values for frames where
        the identity does not exist. These get filtered out when filtering out
        unlabeled frames, since those frames are always unlabeled.
        :return: returns per frame features in dictionary with this form

        {
            'pairwise_distances': 2D numpy array,
            'angles': 2D numpy array,
            'point_speeds': 2D numpy array
        }
        """

        if labels is None:
            features = self._per_frame

        else:
            # return only features for labeled frames
            features = {
                k: v[labels != src.project.track_labels.TrackLabels.Label.NONE, ...]
                for k, v in self._per_frame.items()
            }

        return {
            feature_name: features[feature_name]
            for feature_name in self.get_feature_names(
                use_social, self._extended_features)
        }

    def get_features(self, window_size: int, use_social: bool):
        """
        get features and corresponding frame indexes for classification
        omits frames where the identity is not valid, so 'frame_indexes' array
        may not be consecutive
        :param window_size: window size to use
        :param use_social: if true, include social features in returned data
        No effect for v2 pose files.
        :return: dictionary with the following keys:

          'per_frame': dict with feature name as keys, numpy array as values
          'window': dict, see _compute_window_features
          'frame_indexes': 1D np array, maps elements of per frame and window
             feature arrays back to global frame indexes
        """
        window_features = self.get_window_features(window_size, use_social)

        per_frame = {}
        indexes = np.arange(self._num_frames)[self._frame_valid == 1]

        all_features = self.get_feature_names(
            use_social, self._extended_features)

        for feature in all_features:
            per_frame[feature] = self._per_frame[feature][
                self._frame_valid == 1, ...]

        window = {}
        for key in window_features:
            window[key] = {}
            for op in window_features[key]:
                window[key][op] = window_features[key][op][self._frame_valid == 1]

        return {
            'per_frame': per_frame,
            'window': window,
            'frame_indexes': indexes
        }

    def __compute_window_features(self, window_size: int):
        """
        compute all window features using a given window size
        :param window_size: number of frames on each side of the current frame
        to include in the window
        (so, for example, if window_size = 5, then the total number of frames
        in the window is 11)
        :return: dictionary of the form:
        {
            'angles' {
                'mean': numpy float32 array with shape (#frames, #angles),
                'std_dev': numpy float32 array with shape (#frames, #angles),
            },
            'pairwise_distances' {
                'mean': numpy float32 array with shape (#frames, #distances),
                'median': numpy float32 array with shape (#frames, #distances),
                'std_dev': numpy float32 array with shape (#frames, #distances),
                'max': numpy float32 array with shape (#frames, #distances),
                'min' numpy float32 array with shape (#frames, #distances),
            },
            'point_speeds': {...},
            ...
        }
        """

        window_features = {}

        for key in self._feature_modules:
            window_features.update(
                self._feature_modules[key].window(self._identity, window_size,
                                                  self._per_frame))

        return window_features

    def get_feature_column_names(self, use_social: bool, ):
        """
        build up a list of column names for the 2D feature array that will be
        passed to the classifier
        """

        column_names = []
        # start with point_mask as a special case, as it is not computed in
        # a feature module -- it's just added directly from self._pose_est
        per_frame_features = {'point_mask': [f'{point.name } point mask' for point in PoseEstimation.KeypointIndex]}
        window_features = {}
        base_groups = [m.name() for m in _FEATURE_MODULES]
        for key in self._feature_modules:
            # handle base (& social) features
            if key in base_groups:
                if not use_social and key == SocialFeatureGroup.name():
                    continue
                per_frame_features.update(
                    self._feature_modules[key].feature_names())
                window_features.update(
                    self._feature_modules[key].window_feature_names())
            else:
                # handle extended features
                if self._extended_features is None:
                    per_frame_features.update(
                        self._feature_modules[key].feature_names())
                    window_features.update(
                        self._feature_modules[key].window_feature_names())
                elif key in self._extended_features:
                    per_frame_features.update(
                        self._feature_modules[key].feature_names(self._extended_features[key]))
                    window_features.update(
                        self._feature_modules[key].window_feature_names(self._extended_features[key]))

        # generate a list of column names in the same order the data is
        # assembled in Classifier.combine_data()

        # first, iterate over the per frame feature names alphabetically
        for f in sorted(per_frame_features):
            column_names += per_frame_features[f]

        for f in sorted(window_features):
            for col in sorted(window_features[f]):
                for op in sorted(window_features[f][col]):
                    column_names.append(f"{op} {col}")
        return column_names

    @classmethod
    def merge_per_frame_features(cls, features: dict, include_social: bool,
                                 extended_features=None):
        """
        merge a list of per-frame features where each element in the list is
        a set of per-frame features computed for an individual animal
        :param features: list of per-frame feature instances
        :param include_social:
        :param extended_features: optional extended feature configuration,
        dict with feature groups as keys, lists of feature names as
        values. If None, all extended features are enabled.

        :return: dict of the form
        {
            'pairwise_distances':,
            'angles':,
            'point_speeds':,
            'point_masks':
        }
        """

        # determine which features are in common between total feature set and
        # available computed features
        feature_intersection = cls.get_feature_names(
            include_social, extended_features)

        for feature_dict in features:
            feature_intersection &= set(feature_dict.keys())

        return {
            feature_name: np.concatenate([x[feature_name] for x in features])
            for feature_name in feature_intersection
        }

    @classmethod
    def merge_window_features(
            cls,
            features: dict,
            include_social: bool,
            extended_features: typing.Optional[typing.Dict] = None
    ):
        """
        merge a list of window features where each element in the list is the
        set of window features computed for an individual animal
        :param features: list of window feature instances
        :param include_social:
        :param extended_features: optional extended feature configuration,
        dict with feature groups as keys, lists of feature module names as
        values. If None, all extended features are enabled.
        :return: dictionary of the form:
        {
            'angles' {
                'mean': numpy float32 array with shape (#frames, #angles),
                'std_dev': numpy float32 array with shape (#frames, #angles)
            },
            'pairwise_distances' {
                'mean': numpy float32 array with shape (#frames, #distances),
                'median': numpy float32 array with shape (#frames, #distances),
                'std_dev': numpy float32 array with shape (#frames, #distances),
                'max': numpy float32 array with shape (#frames, #distances),
                'min' numpy float32 array with shape (#frames, #distances),
            },
            'point_speeds' {
                ...
            }
        }
        """
        merged = {}

        # determine which features are in common between total feature set and
        # available computed features
        feature_intersection = cls.get_feature_names(
            include_social, extended_features)

        for feature_dict in features:
            feature_intersection &= set(feature_dict.keys())

        for feature_name in feature_intersection:
            merged[feature_name] = {}
            # grab list of operations for this window feature from the first
            # dictionary in features
            window_ops = features[0][feature_name].keys()
            for f in features:
                assert (set(window_ops) == set(f[feature_name].keys()))
            for op in window_ops:
                merged[feature_name][op] = np.concatenate(
                    [x[feature_name][op] for x in features])

        return merged

    @classmethod
    def get_feature_names(
            cls, include_social: bool,
            extended_features: typing.Optional[
                typing.Dict[str, typing.List[str]]] = None
    ) -> typing.Set[str]:
        """
        get all the feature module names
        :param include_social: if true, include all social features
        :param extended_features: if None, include all extended features,
        otherwise only include those specified
        :return: set of feature module names
        """
        module_names = {'point_mask'}
        for m in _FEATURE_MODULES:
            if not include_social and m is SocialFeatureGroup:
                continue
            module_names.update(m.module_names())
        if extended_features is None:
            for m in _EXTENDED_FEATURE_MODULES:
                module_names.update(m.module_names())
        else:
            for _, features in extended_features.items():
                module_names.update(features)
        return module_names

    @classmethod
    def get_available_extended_features(
            cls,
            pose_version: int,
            static_objects: typing.List[str]
    ) -> typing.Dict[str, typing.List[str]]:
        """
        get all of the extended features that can be used given a minimum pose
        version and list of available static objects
        :param pose_version: integer pose version
        :param static_objects: list of static object names
        :return: dictionary of supported extended features, where the keys
        are "feature group" name(s) and values are feature names that can
        be used from that group

        TODO social features probably should get moved into the 'extended'
          features, but they are still handled as a special case
        """

        return {
            feature_group.name(): feature_group.get_supported_feature_modules(
                pose_version, static_objects) for feature_group in
            _EXTENDED_FEATURE_MODULES
        }
