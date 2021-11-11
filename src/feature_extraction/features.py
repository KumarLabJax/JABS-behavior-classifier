from pathlib import Path

import h5py
import numpy as np
import scipy.stats

import src.project.track_labels
from src.feature_extraction.angle_index import AngleIndex
from src.utils.utilities import n_choose_r
from src.pose_estimation import PoseEstimation, PoseHashException

# import feature modules
from .base_features import BaseFeatureGroup
from .social_features import SocialFeatureGroup

FEATURE_VERSION = 3


class FeatureVersionException(Exception):
    pass


class DistanceScaleException(Exception):
    pass


class IdentityFeatures:
    """
    per frame and window features for a single identity
    """

    # For social interaction we will consider a subset
    # of points to capture just the most important
    # information for social.
    _social_point_subset = [
        PoseEstimation.KeypointIndex.NOSE,
        PoseEstimation.KeypointIndex.BASE_NECK,
        PoseEstimation.KeypointIndex.BASE_TAIL,
    ]

    _version = FEATURE_VERSION

    # The operations that are performed on windows of per-frame features to
    # generate the window features. This is organized in a dictionary where the
    # key is the operation name and the value is a function that takes an numpy
    # array of values and returns a single feature value.
    _window_feature_operations = {
        "mean": np.ma.mean,
        "median": np.ma.median,
        "std_dev": np.ma.std,
        "max": np.ma.amax,
        "min": np.ma.amin
    }

    # window feature operations used for angle based features
    _window_feature_operations_circular = {
        "mean": lambda x: scipy.stats.circmean(x, high=360),
        "std_dev": lambda x: scipy.stats.circstd(x, high=360),
    }

    # window feature operations used for angle based features
    _window_feature_operations_circular_2 = {
        "mean": lambda x: scipy.stats.circmean(x, low=-180, high=180),
        "std_dev": lambda x: scipy.stats.circstd(x, low=-180, high=180),
    }

    _per_frame_features = [
        'angles',
        'pairwise_distances',
        'point_speeds',
        'point_mask',
        'angular_velocity',
        'centroid_velocity_mag',
        'centroid_velocity_dir',
        'nose_velocity_mag',
        'nose_velocity_dir',
        'base_tail_velocity_mag',
        'base_tail_velocity_dir',
        'left_front_paw_velocity_mag',
        'left_front_paw_velocity_dir',
        'right_front_paw_velocity_mag',
        'right_front_paw_velocity_dir'
    ]

    _per_frame_social_features = [
        'closest_distances',
        'closest_fov_distances',
        'closest_fov_angles',
        'social_pairwise_distances',
        'social_pairwise_fov_distances',
    ]

    _window_features = [
        'angles',
        'pairwise_distances',
        'point_speeds',
        'angular_velocity',
        'centroid_velocity_mag',
        'centroid_velocity_dir',
        'nose_velocity_mag',
        'nose_velocity_dir',
        'base_tail_velocity_mag',
        'base_tail_velocity_dir',
        'left_front_paw_velocity_mag',
        'left_front_paw_velocity_dir',
        'right_front_paw_velocity_mag',
        'right_front_paw_velocity_dir'
    ]

    _window_social_features = [
        'closest_distances',
        'closest_fov_distances',
        'closest_fov_angles',
        'social_pairwise_distances',
        'social_pairwise_fov_distances',
    ]

    _circular_features = ['angles']
    _circular_features_2 = ['centroid_velocity_dir', 'nose_velocity_dir',
                            'closest_fov_angles', 'base_tail_velocity_dir',
                            'left_front_paw_velocity_dir',
                            'right_front_paw_velocity_dir']

    # TODO  For now this is taken from the ICY paper where the full field of
    # view is 240 degrees. Do we want this to be configurable?
    half_fov_deg = 120

    def __init__(self, source_file, identity, directory, pose_est, force=False,
                 fps=30, distance_scale_factor: float = 1.0):
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
        """

        self._num_frames = pose_est.num_frames
        self._fps = fps
        self._pose_hash = pose_est.hash
        self._identity = identity

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

        self._feature_modules = [
            BaseFeatureGroup(pose_est, self._distance_scale_factor)
        ]
        if self._compute_social_features:
            self._feature_modules.append(
                SocialFeatureGroup(pose_est, self._distance_scale_factor))


        # will hold an array that indicates if each frame is valid for this
        # identity or not
        self._frame_valid = None

        # per frame features
        self._per_frame = {
            'point_mask': pose_est.get_identity_point_mask(identity)
        }

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

        for feature_module in self._feature_modules:
            self._per_frame.update(feature_module.per_frame(self._identity))

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
            for feature in self._per_frame_features:
                if feature in ['point_mask']:
                    continue
                self._per_frame[feature] = feature_grp[feature][:]

            assert len(self._frame_valid) == self._num_frames
            assert len(self._per_frame['pairwise_distances']) == self._num_frames
            assert len(self._per_frame['angles']) == self._num_frames

            if self._compute_social_features:
                self._closest_identities = features_h5['closest_identities'][:]
                self._closest_fov_identities = features_h5['closest_fov_identities'][:]

                for feature in self._per_frame_social_features:
                    self._per_frame[feature] = feature_grp[feature][:]
                    assert self._per_frame[feature].shape[0] == self._num_frames

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
            for feature in self._per_frame_features:
                # point mask is obtained from the pose file, don't save it
                if feature in ['point_mask']:
                    continue
                grp.create_dataset(feature, data=self._per_frame[feature])

            if self._compute_social_features:
                features_h5['closest_identities'] = self._closest_identities
                features_h5['closest_fov_identities'] = self._closest_fov_identities

                for feature in self._per_frame_social_features:
                    grp[feature] = self._per_frame[feature]

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

            for feature in self._window_features:
                for op in features[feature]:
                    grp.create_dataset(f'{feature}/{op}',
                                       data=features[feature][op])

            if self._compute_social_features:
                for feature_name in self._window_social_features:
                    for op in features[feature_name]:
                        grp.create_dataset(f'{feature_name}/{op}',
                                           data=features[feature_name][op])

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

            for feature_name in self._window_features:
                window_features[feature_name] = {}
                for op in feature_grp[feature_name].keys():
                    window_features[feature_name][op] = \
                        feature_grp[f'{feature_name}/{op}'][:]
                    assert len(
                        window_features[feature_name][op]) == self._num_frames

            if self._compute_social_features:
                for feature_name in self._window_social_features:
                    window_features[feature_name] = {}
                    for op in feature_grp[feature_name].keys():
                        window_features[feature_name][op] = feature_grp[f'{feature_name}/{op}'][:]
                        assert len(window_features[feature_name][op]) == self._num_frames

            return window_features

    def get_window_features(self, window_size, use_social: bool, labels=None,
                            force=False):
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

        if use_social:
            feature_intersection = set(self._window_features + self._window_social_features)
        else:
            feature_intersection = set(self._window_features)

        #feature_intersection &= set(features.keys())

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

        if use_social:
            feature_intersection = set(self._per_frame_features + self._per_frame_social_features)
        else:
            feature_intersection = set(self._per_frame_features)

        feature_intersection &= set(self._per_frame.keys())

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
            for feature_name in feature_intersection
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

        if self._compute_social_features and use_social:
            all_features = self._per_frame_features + self._per_frame_social_features
        else:
            all_features = self._per_frame_features

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

    def __compute_window_features(self, window_size):
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

        for feature_module in self._feature_modules:
            window_features.update(feature_module.window(self._identity, window_size, self._per_frame))

        return window_features

    @classmethod
    def get_feature_names(cls, include_social_features):
        """
        return list of human readable feature names, starting with per frame
        features followed by window features. feature names in each group are
        sorted

        NOTE: the order of the list that is returned must match the order of
        the columns in the nframes x nfeatures arrays passed to the training and
        classification functions. This needs to sort the features the same way
        as Classifier.combine_data()
        :return: list of human readable feature names
        """
        feature_list = []

        if include_social_features:
            full_per_frame_features = cls._per_frame_features + cls._per_frame_social_features
        else:
            full_per_frame_features = cls._per_frame_features

        for feature in sorted(full_per_frame_features):
            if feature == 'angles':
                feature_list.extend([
                    f"angle {AngleIndex.get_angle_name(angle)}" for angle in
                    AngleIndex])
            elif feature == 'pairwise_distances':
                feature_list.extend(IdentityFeatures.get_distance_names())
            elif feature == 'point_speeds':
                feature_list.extend([
                    f"{p.name} speed" for p in PoseEstimation.KeypointIndex])
            elif feature == 'closest_distances':
                feature_list.append("closest social distance")
            elif feature == 'closest_fov_distances':
                feature_list.append("closest social distance in FoV")
            elif feature == 'closest_fov_angles':
                feature_list.append("angle of closest social distance in FoV")
            elif feature == 'social_pairwise_distances':
                feature_list.extend([
                    f"social dist. {sdn}"
                    for sdn in IdentityFeatures.get_social_distance_names()])
            elif feature == 'social_pairwise_fov_distances':
                feature_list.extend([
                    f"social fov dist. {sdn}"
                    for sdn in IdentityFeatures.get_social_distance_names()])
            elif feature == 'point_mask':
                feature_list.extend([
                    f"{p.name} point mask" for p in PoseEstimation.KeypointIndex
                ])
            else:
                feature_list.append(feature)

        if include_social_features:
            full_window_features = cls._window_features + cls._window_social_features
        else:
            full_window_features = cls._window_features

        for feature in sorted(full_window_features):
            # [source_feature_name][operator_applied] : numpy array
            # iterate over operator names
            if feature in cls._circular_features + cls._circular_features_2:
                for op in sorted(cls._window_feature_operations_circular):
                    if feature == 'angles':
                        feature_list.extend(
                            [f"{op} angle {AngleIndex.get_angle_name(angle)}" for angle in
                             AngleIndex])
                    elif feature == 'closest_fov_angles':
                        feature_list.append(f"{op} angle of closest social distance in FoV")
                    elif feature == 'centroid_velocity_dir':
                        feature_list.append(f"{op} centroid velocity orientation")
                    elif feature == 'nose_velocity_dir':
                        feature_list.append(f"{op} nose velocity orientation")
                    else:
                        feature_list.append(f"{op} {feature}")
            else:
                for op in sorted(cls._window_feature_operations):
                    if feature == 'pairwise_distances':
                        feature_list.extend(
                            [f"{op} {d}" for d in IdentityFeatures.get_distance_names()])
                    elif feature == 'point_speeds':
                        feature_list.extend([
                            f"{op} {p.name} speed" for p in
                            PoseEstimation.KeypointIndex])
                    elif feature == 'closest_distances':
                        feature_list.append(f"{op} closest social distance")
                    elif feature == 'closest_fov_distances':
                        feature_list.append(f"{op} closest social distance in FoV")
                    elif feature == 'social_pairwise_distances':
                        feature_list.extend([
                            f"{op} social dist. {sdn}"
                            for sdn in IdentityFeatures.get_social_distance_names()])
                    elif feature == 'social_pairwise_fov_distances':
                        feature_list.extend([
                            f"{op} social fov dist. {sdn}"
                            for sdn in IdentityFeatures.get_social_distance_names()])
                    elif feature == 'centroid_velocity_mag':
                        feature_list.append(f"{op} centroid velocity magnitude")
                    else:
                        feature_list.append(f"{op} {feature}")

        return feature_list

    @classmethod
    def merge_per_frame_features(cls, features, include_social):
        """
        merge a list of per-frame features where each element in the list is
        a set of per-frame features computed for an individual animal
        :param features: list of per-frame feature instances
        :param include_social:
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
        if include_social:
            feature_intersection = set(cls._per_frame_features + cls._per_frame_social_features)
        else:
            feature_intersection = set(cls._per_frame_features)

        for feature_dict in features:
            feature_intersection &= set(feature_dict.keys())

        return {
            feature_name: np.concatenate([x[feature_name] for x in features])
            for feature_name in feature_intersection
        }

    @classmethod
    def merge_window_features(cls, features, include_social):
        """
        merge a list of window features where each element in the list is the
        set of window features computed for an individual animal
        :param features: list of window feature instances
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
        if include_social:
            feature_intersection = set(
                cls._window_features
                + cls._window_social_features
            )
        else:
            feature_intersection = set(cls._window_features)

        for feature_dict in features:
            feature_intersection &= set(feature_dict.keys())

        for feature_name in feature_intersection:
            merged[feature_name] = {}
            if feature_name in cls._circular_features + cls._circular_features_2:
                operations = cls._window_feature_operations_circular
            else:
                operations = cls._window_feature_operations
            for op in operations:
                merged[feature_name][op] = np.concatenate(
                    [x[feature_name][op] for x in features])

        return merged

    @staticmethod
    def get_distance_names():
        """
        get list of human readable names for each value computed by
        _compute_pairwise_distances
        :return: list of distance names where each is a string of the form
        "distance_name_1-distance_name_2"
        """
        distances = []
        point_names = [p.name for p in PoseEstimation.KeypointIndex]
        for i in range(0, len(point_names)):
            p1 = point_names[i]
            for p2 in point_names[i + 1:]:
                distances.append(f"{p1}-{p2}")
        return distances

    @classmethod
    def get_social_distance_names(cls):
        """
        get list of human readable names for each value computed by
        _compute_social_pairwise_distance
        :return: list of distance names where each is a string of the form
        "distance_name_1-distance_name_2"
        """
        dist_names = []
        for kpi1 in cls._social_point_subset:
            for kpi2 in cls._social_point_subset:
                dist_names.append(f"{kpi1.name}-{kpi2.name}")
        return dist_names

