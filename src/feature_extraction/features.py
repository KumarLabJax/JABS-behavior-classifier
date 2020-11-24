import enum
import math
from pathlib import Path

import h5py
import numpy as np
import scipy.stats

from src.labeler.track_labels import TrackLabels
from src.pose_estimation import PoseEstimationV3


def n_choose_r(n, r):
    """
    compute number of unique selections (disregarding order) of r items from
    a set of n items
    :param n: number of elements to select from
    :param r: number of elements to select
    :return: total number of combinations disregarding order
    """
    def fact(v):
        res = 1
        for i in range(2, v + 1):
            res = res * i
        return res
    return fact(n) // (fact(r) * fact(n - r))


class AngleIndex(enum.IntEnum):
    """ enum defining the indexes of the angle features """
    NOSE_BASE_NECK_RIGHT_FRONT_PAW = 0
    NOSE_BASE_NECK_LEFT_FRONT_PAW = 1
    RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE = 2
    LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE = 3
    BASE_NECK_CENTER_SPINE_BASE_TAIL = 4
    RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE = 5
    LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE = 6
    RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL = 7
    LEFT_REAR_PAW_BASE_TAIL_MID_TAIL = 8
    CENTER_SPINE_BASE_TAIL_MID_TAIL = 9
    BASE_TAIL_MID_TAIL_TIP_TAIL = 10


class IdentityFeatures:
    """
    per frame and window features for a single identity
    """

    # For social interaction we will consider a subset
    # of points to capture just the most important
    # information for social.
    _social_point_subset = [
        PoseEstimationV3.KeypointIndex.NOSE,
        PoseEstimationV3.KeypointIndex.BASE_NECK,
        PoseEstimationV3.KeypointIndex.BASE_TAIL,
    ]

    _num_distances = n_choose_r(len(PoseEstimationV3.KeypointIndex), 2)
    _num_social_distances = len(_social_point_subset) ** 2
    _num_angles = len(AngleIndex)
    _version = "1.0.0"

    # The operations that are performed on windows of per-frame features to
    # generate the window features. This is organized in a dictionary where the
    # key is the operation name and the value is a function that takes an numpy
    # array of values and returns a single feature value.
    _window_feature_operations = {
        "mean": lambda x: np.mean(x),
        "median": lambda x: np.median(x),
        "std_dev": lambda x: np.std(x),
        "max": lambda x: np.amax(x),
        "min": lambda x: np.amin(x)
    }

    # a copy of the above operations, but used for values that are circular
    # (e.g. angles)
    _window_feature_operations_circular = {
        "mean": lambda x: scipy.stats.circmean(np.radians(x)),
        "median": lambda x: np.median(x), #scipy.stats does not have a circular version of median
        "std_dev": lambda x: scipy.stats.circstd(np.radians(x)),
        "max": lambda x: np.amax(x),
        "min": lambda x: np.amin(x)
    }

    # keys should match in the above dicts
    assert(_window_feature_operations.keys() ==
           _window_feature_operations_circular.keys())

    _per_frame_features = [
        'angles',
        'pairwise_distances',
        'point_speeds',
        'point_mask'
    ]

    _per_frame_social_features = [
        'closest_distances',
        'closest_fov_distances',
        'closest_fov_angles',
        'social_pairwise_distances',
        'social_pairwise_fov_distances',
    ]

    _window_features = [
        'percent_frames_present',
        'angles',
        'pairwise_distances',
        'point_speeds'
    ]

    _window_social_features = [
        'closest_distances',
        'closest_fov_distances',
        'closest_fov_angles',
        'social_pairwise_distances',
        'social_pairwise_fov_distances',
    ]

    # TODO  For now this is taken from the ICY paper where the full field of
    # view is 240 degrees. Do we want this to be configurable?
    half_fov_deg = 120

    def __init__(self, video_name, identity, directory, pose_est, force=False):
        """
        :param video_name: name of the video file, used for generating filenames
        for saving extracted features into the project directory
        :param identity: identity to extract features for
        :param directory: path of the project directory
        :param pose_est: PoseEstimationV3 object corresponding to this video
        :param force: force regeneration of per frame features even if the
        per frame feature .h5 file exists for this video/identity
        """

        self._video_name = Path(video_name)
        self._num_frames = pose_est.num_frames
        self._identity = identity
        self._project_feature_directory = Path(directory)
        self._identity_feature_dir = (
                self._project_feature_directory /
                self._video_name.stem /
                str(self._identity)
        )
        self._include_social_features = pose_est.format_major_version >= 3
        if self._include_social_features:
            self._closest_identities = np.full(self._num_frames, -1,
                                               dtype=np.int16)
            self._closest_fov_identities = np.full(self._num_frames, -1,
                                                   dtype=np.int16)
        else:
            self._closest_identities = None
            self._closest_fov_identities = None

        # will hold an array that indicates if each frame is valid for this
        # identity or not
        self._frame_valid = None

        # per frame features
        self._per_frame = {}
        for feature in self._per_frame_features:
            if feature == 'pairwise_distances':
                self._per_frame[feature] = np.empty(
                    (self._num_frames, self._num_distances), dtype=np.float32)
            elif feature == 'angles':
                self._per_frame[feature] = np.empty(
                    (self._num_frames, self._num_angles), dtype=np.float32)
            elif feature == 'point_speeds':
                self._per_frame[feature] = np.empty(
                    (self._num_frames, len(PoseEstimationV3.KeypointIndex)),
                    dtype=np.float32)
            elif feature == 'point_mask':
                self._per_frame[feature] = \
                    pose_est.get_identity_point_mask(identity)
            else:
                raise ValueError(
                    f"Missing feature initialization for: {feature}")

        if self._include_social_features:
            for feature in self._per_frame_social_features:
                if feature in ('social_pairwise_distances',
                               'social_pairwise_fov_distances'):
                    self._per_frame[feature] = np.zeros(
                        (self._num_frames, self._num_social_distances),
                        dtype=np.float32)
                else:
                    self._per_frame[feature] = np.zeros(
                        self._num_frames,
                        dtype=np.float32)

        if force:
            self.__initialize_from_pose_estimation(pose_est)
        else:
            try:
                # try to load from an h5 file if it exists
                self.__load_from_file()
            except OSError:
                # otherwise compute the per frame features and save
                self.__initialize_from_pose_estimation(pose_est)

    @property
    def window_ops(self):
        """
        get list of window operation names
        :return: list of strings containing window operation names
        """
        return self._window_feature_operations.keys()

    def __initialize_from_pose_estimation(self, pose_est):
        """
        Initialize from a PoseEstimation object and save them in an h5 file

        :param pose_est: PoseEstimation object used to initialize self
        :return: None
        """

        idx = PoseEstimationV3.KeypointIndex

        for frame in range(pose_est.num_frames):
            points, mask = pose_est.get_points(frame, self._identity)

            if points is not None:
                self._per_frame['pairwise_distances'][frame] = self._compute_pairwise_distance(points)
                self._per_frame['angles'][frame] = self._compute_angles(points)

                if self._include_social_features:
                    # Find the distance and identity of the closest animal at each frame, as well
                    # as the distance, identity and angle of the closes animal in field of view.
                    # In order to calculate this we require that both animals have a valid
                    # convex hull and the the self identity has a valid nose point and
                    # base neck point (which is used to calculate FoV).
                    self_shape = pose_est.get_identity_convex_hulls(self._identity)[frame]
                    if self_shape is not None and mask[idx.NOSE] == 1 and mask[idx.BASE_NECK] == 1:
                        closest_dist = None
                        closest_fov_dist = None
                        for curr_id in pose_est.identities:
                            if curr_id != self._identity:
                                other_shape = pose_est.get_identity_convex_hulls(curr_id)[frame]

                                if other_shape is not None:
                                    curr_dist = self_shape.distance(other_shape)
                                    if closest_dist is None or curr_dist < closest_dist:
                                        self._closest_identities[frame] = curr_id
                                        self._per_frame['closest_distances'][frame] = curr_dist
                                        closest_dist = curr_dist

                                    self_base_neck_point = points[idx.BASE_NECK, :]
                                    self_nose_point = points[idx.NOSE, :]
                                    other_centroid = np.array(other_shape.centroid)

                                    view_angle = self.compute_angle(
                                        self_nose_point,
                                        self_base_neck_point,
                                        other_centroid)

                                    # for FoV we want the range of view angle to be [180, -180)
                                    if view_angle > 180:
                                        view_angle -= 360
                                    
                                    if abs(view_angle) <= self.half_fov_deg:
                                        # other animal is in FoV
                                        if closest_fov_dist is None or curr_dist < closest_fov_dist:
                                            self._closest_fov_identities[frame] = curr_id
                                            self._per_frame['closest_fov_distances'][frame] = curr_dist
                                            self._per_frame['closest_fov_angles'][frame] = view_angle
                                            closest_fov_dist = curr_dist

                        if self._closest_identities[frame] != -1:
                            closest_points, _ = pose_est.get_points(
                                frame, self._closest_identities[frame])
                            social_pt_indexes = [idx.value for idx in self._social_point_subset]
                            social_pairwise_distances = self._compute_social_pairwise_distance(
                                points[social_pt_indexes, ...],
                                closest_points[social_pt_indexes, ...])
                            self._per_frame['social_pairwise_distances'][frame] = social_pairwise_distances

                        if self._closest_fov_identities[frame] != -1:
                            closest_points, _ = pose_est.get_points(
                                frame, self._closest_fov_identities[frame])
                            social_pt_indexes = [idx.value for idx in self._social_point_subset]
                            social_pairwise_distances = self._compute_social_pairwise_distance(
                                points[social_pt_indexes, ...],
                                closest_points[social_pt_indexes, ...])
                            self._per_frame['social_pairwise_fov_distances'][frame] = social_pairwise_distances

        # indicate this identity exists in this frame
        self._frame_valid = pose_est.identity_mask(self._identity)

        self._per_frame['point_speeds'] = self._compute_point_speeds(
            *pose_est.get_identity_poses(self._identity))

        self.save_per_frame()

    def __load_from_file(self):
        """
        initialize from state previously saved in a h5 file on disk
        :return: None
        """

        path = self._identity_feature_dir / 'per_frame.h5'

        with h5py.File(path, 'r') as features_h5:
            feature_grp = features_h5['features']

            # load per frame features
            self._frame_valid = features_h5['frame_valid'][:]
            self._per_frame['pairwise_distances'] = feature_grp['pairwise_distances'][:]
            self._per_frame['angles'] = feature_grp['angles'][:]
            self._per_frame['point_speeds'] = feature_grp['point_speeds'][:]

            assert len(self._frame_valid) == self._num_frames
            assert len(self._per_frame['pairwise_distances']) == self._num_frames
            assert len(self._per_frame['angles']) == self._num_frames

            if self._include_social_features:
                self._closest_identities = features_h5['closest_identities'][:]
                self._closest_fov_identities = features_h5['closest_fov_identities'][:]

                for feature in self._per_frame_social_features:
                    self._per_frame[feature] = feature_grp[feature][:]
                    assert self._per_frame[feature].shape[0] == self._num_frames

    def save_per_frame(self):
        """ save per frame features to a h5 file """

        self._identity_feature_dir.mkdir(mode=0o775, exist_ok=True, parents=True)

        file_path = self._identity_feature_dir / 'per_frame.h5'

        with h5py.File(file_path, 'w') as features_h5:
            features_h5.attrs['num_frames'] = self._num_frames
            features_h5.attrs['identity'] = self._identity
            features_h5.attrs['version'] = self._version
            features_h5.create_dataset('frame_valid', data=self._frame_valid)

            grp = features_h5.create_group('features')
            grp.create_dataset('pairwise_distances', data=self._per_frame['pairwise_distances'])
            grp.create_dataset('angles', data=self._per_frame['angles'])
            grp.create_dataset('point_speeds', data=self._per_frame['point_speeds'])

            if self._include_social_features:
                features_h5['closest_identities'] = self._closest_identities
                features_h5['closest_fov_identities'] = self._closest_fov_identities

                for feature in self._per_frame_social_features:
                    grp[feature] = self._per_frame[feature]

    def save_window_features(self, features, radius):
        """
        save window features to an h5 file
        :param features: window features to save
        :param radius:
        :return: None
        """
        path = self._identity_feature_dir / f"window_features_{radius}.h5"

        with h5py.File(path, 'w') as features_h5:
            features_h5.attrs['num_frames'] = self._num_frames
            features_h5.attrs['identity'] = self._identity
            features_h5.attrs['version'] = self._version
            features_h5.attrs['radius'] = radius

            grp = features_h5.create_group('features')

            for feature in self._window_features:
                if feature == 'percent_frames_present':
                    grp.create_dataset(feature, data=features[feature])
                else:
                    for op in features[feature]:
                        grp.create_dataset(f'{feature}/{op}',
                                           data=features[feature][op])

            if self._include_social_features:
                for feature_name in self._window_social_features:
                    for op in features[feature_name]:
                        grp.create_dataset(f'{feature_name}/{op}',
                                        data=features[feature_name][op])

    def _load_window_features(self, radius):
        """
        load window features from an h5 file
        :param radius: window size specified as the number of frames on each
        side of current frame to include in the window
        :return:
        """
        path = self._identity_feature_dir / f"window_features_{radius}.h5"

        window_features = {}

        with h5py.File(path, 'r') as features_h5:
            # TODO verify file version
            assert features_h5.attrs['num_frames'] == self._num_frames
            assert features_h5.attrs['identity'] == self._identity
            assert features_h5.attrs['radius'] == radius

            feature_grp = features_h5['features']

            for feature_name in self._window_features:
                if feature_name == 'percent_frames_present':
                    window_features[feature_name] = feature_grp[feature_name][:]
                    assert len(window_features[feature_name]) == self._num_frames
                else:
                    window_features[feature_name] = {}
                    for op in feature_grp[feature_name].keys():
                        window_features[feature_name][op] = \
                            feature_grp[f'{feature_name}/{op}'][:]
                        assert len(
                            window_features[feature_name][op]) == self._num_frames

            if self._include_social_features:
                for feature_name in self._window_social_features:
                    window_features[feature_name] = {}
                    for op in feature_grp[feature_name].keys():
                        window_features[feature_name][op] = feature_grp[f'{feature_name}/{op}'][:]
                        assert len(window_features[feature_name][op]) == self._num_frames

            return window_features

    def get_window_features(self, radius, labels=None, force=False):
        """
        get window features for a given window size, computing if not previously
        computed and saved as h5 file
        :param radius: number of frames on each side of the current frame to
        include in the window
        :param labels: optional frame labels, if present then only features for
        labeled frames will be returned
        :param force: force regeneration of the window features even if the
        h5 file already exists
        :return: window features for given window size. the format is documented
        in the docstring for _compute_window_features
        """

        if force:
            features = self._compute_window_features(radius)
            self.save_window_features(features, radius)
        else:

            try:
                # h5 file exists for this window size, load it
                features = self._load_window_features(radius)
            except OSError:
                # h5 file does not exist for this window size. compute the features
                # and return after saving
                features = self._compute_window_features(radius)
                self.save_window_features(features, radius)

        if labels is None:
            return features
        else:
            # return only features for labeled frames
            filtered_features = {}

            for key in features:
                if key == 'percent_frames_present':
                    filtered_features[key] = features[key][labels != TrackLabels.Label.NONE]
                else:
                    filtered_features[key] = {}
                    for op in features[key]:
                        filtered_features[key][op] = features[key][op][labels != TrackLabels.Label.NONE]

            return filtered_features

    def get_per_frame(self, labels=None):
        """
        get per frame features
        :param labels: if present, only return features for labeled frames
        :return: returns per frame features in dictionary with this form

        {
            'pairwise_distances': 2D numpy array,
            'angles': 2D numpy array,
            'point_speeds': 2D numpy array
        }
        """
        if labels is None:
            return self._per_frame

        else:
            # return only features for labeled frames
            return {
                k: v[labels != TrackLabels.Label.NONE, ...]
                for k, v in self._per_frame.items()
            }

    def get_unlabeled_features(self, radius, labels):
        """
        get features and corresponding frame indexes for unlabeled frames for
        classification
        :param radius:
        :param labels:
        :return:
        """
        window_features = self.get_window_features(radius)
        filter = np.logical_and(self._frame_valid, labels == TrackLabels.Label.NONE)

        per_frame = {}
        indexes = np.arange(self._num_frames)[filter]
        
        if self._include_social_features:
            all_features = self._per_frame_features + self._per_frame_social_features
        else:
            all_features = self._per_frame_features

        for feature in all_features:
            per_frame[feature] = self._per_frame[feature][
                                 filter, ...]

        window = {}
        for key in window_features:
            if key == 'radius':
                window[key] = window_features[key]
            elif key == 'percent_frames_present':
                window[key] = window_features[key][filter]
            else:
                window[key] = {}
                for op in window_features[key]:
                    window[key][op] = window_features[key][op][filter]

        return {
            'per_frame': per_frame,
            'window': window,
            'frame_indexes': indexes
        }

    def _compute_window_features(self, radius):
        """
        compute all window features using a given window size
        :param radius: number of frames on each side of the current frame to
        include in the window
        :return: dictionary of the form:
        {
            'angles' {
                'mean': numpy float32 array with shape (#frames, #angles),
                'median': numpy float32 array with shape (#frames, #angles),
                'std_dev': numpy float32 array with shape (#frames, #angles),
                'max': numpy float32 array with shape (#frames, #angles),
                'min' numpy float32 array with shape (#frames, #angles),
            },
            'pairwise_distances' {
                'mean': numpy float32 array with shape (#frames, #distances),
                'median': numpy float32 array with shape (#frames, #distances),
                'std_dev': numpy float32 array with shape (#frames, #distances),
                'max': numpy float32 array with shape (#frames, #distances),
                'min' numpy float32 array with shape (#frames, #distances),
            },
            'percent_frames_present': numpy float32 array with shape (#frames,)
        }
        """

        max_window_size = 2 * radius + 1

        window_features = {
            'angles': {},
            'pairwise_distances': {},
            'point_speeds': {},
        }

        # allocate arrays
        for operation in self._window_feature_operations:
            window_features['angles'][operation] = np.empty(
                [self._num_frames, self._num_angles], dtype=np.float32)

        for operation in self._window_feature_operations:
            window_features['pairwise_distances'][operation] = np.empty(
                [self._num_frames, self._num_distances], dtype=np.float32)

        for operation in self._window_feature_operations:
            window_features['point_speeds'][operation] = np.empty(
                [self._num_frames, len(PoseEstimationV3.KeypointIndex)],
                dtype=np.float32)

        window_features['percent_frames_present'] = np.empty((self._num_frames, 1),
                                                             dtype=np.float32)

        # allocate arrays for social
        if self._include_social_features:
            for feature_name in self._window_social_features:
                window_features[feature_name] = {}
                if feature_name in ('social_pairwise_distances', 'social_pairwise_fov_distances'):
                    for operation in self._window_feature_operations:
                        window_features[feature_name][operation] = np.empty(
                            (self._num_frames, self._num_social_distances),
                            dtype=np.float32)
                else:
                    for operation in self._window_feature_operations:
                        window_features[feature_name][operation] = np.empty(
                            self._num_frames, dtype=np.float32)

        # compute window features
        for i in range(self._num_frames):

            # identity doesn't exist for this frame don't bother to compute the
            # window features
            if not self._frame_valid[i]:
                continue

            slice_start = max(0, i - radius)
            slice_end = min(i + radius + 1, self._num_frames)

            frame_valid = self._frame_valid[slice_start:slice_end]
            frames_in_window = np.count_nonzero(frame_valid)

            window_features['percent_frames_present'][i, 0] = frames_in_window / max_window_size

            # compute window features for angles
            for angle_index in range(0, self._num_angles):
                window_values = self._per_frame['angles'][slice_start:slice_end,
                                angle_index][frame_valid == 1]

                for operation in self._window_feature_operations_circular:
                    window_features['angles'][operation][i, angle_index] = \
                        self._window_feature_operations[operation](window_values)

            # compute window features for distances
            for distance_index in range(0, self._num_distances):
                window_values = self._per_frame['pairwise_distances'][
                                    slice_start:slice_end,
                                    distance_index][frame_valid == 1]

                for operation in self._window_feature_operations:
                    window_features['pairwise_distances'][operation][
                        i, distance_index] = self._window_feature_operations[
                        operation](window_values)

            # compute window features for point speeds
            for kp_index_enum in PoseEstimationV3.KeypointIndex:
                window_values = self._per_frame['point_speeds'][
                                    slice_start:slice_end,
                                    kp_index_enum.value][frame_valid == 1]

                for operation in self._window_feature_operations:
                    window_features['point_speeds'][operation][
                        i, kp_index_enum.value] = self._window_feature_operations[
                        operation](window_values)

            # compute social window features using a general approach for
            # 1D and 2D feature shapes
            if self._include_social_features:
                for feature_name in self._window_social_features:
                    window_values = self._per_frame[feature_name][slice_start:slice_end, ...]

                    assert window_values.ndim == 1 or window_values.ndim == 2
                    if window_values.ndim == 1:
                        for op_name, op in self._window_feature_operations.items():
                            window_features[feature_name][op_name][i] = op(
                                window_values[frame_valid == 1])
                    else:
                        for j in range(window_values.shape[1]):
                            if feature_name == 'closest_fov_angles':
                                for op_name, op in self._window_feature_operations_circular.items():
                                    window_features[feature_name][op_name][i, j] = op(
                                        window_values[:, j][frame_valid == 1])
                            else:
                                for op_name, op in self._window_feature_operations.items():
                                    window_features[feature_name][op_name][i, j] = op(
                                        window_values[:, j][frame_valid == 1])

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
        as SklClassifier.combine_data()
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
                    f"angle {angle.name}" for angle in AngleIndex])
            elif feature == 'pairwise_distances':
                feature_list.extend(IdentityFeatures.get_distance_names())
            elif feature == 'point_speeds':
                feature_list.extend([
                    f"{p.name} speed" for p in PoseEstimationV3.KeypointIndex])
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
                    f"{p.name} point mask" for p in PoseEstimationV3.KeypointIndex
                ])
            else:
                feature_list.extend(feature)

        if include_social_features:
            full_window_features = cls._window_features + cls._window_social_features
        else:
            full_window_features = cls._window_features

        for feature in sorted(full_window_features):
            if feature == 'percent_frames_present':
                feature_list.append(feature)
            else:
                # [source_feature_name][operator_applied] : numpy array
                # iterate over operator names
                for op in sorted(cls._window_feature_operations):

                    if feature == 'angles':
                        feature_list.extend(
                            [f"{op} angle {angle.name}" for angle in AngleIndex])
                    elif feature == 'pairwise_distances':
                        feature_list.extend(
                            [f"{op} {d}" for d in IdentityFeatures.get_distance_names()])
                    elif feature == 'point_speeds':
                        feature_list.extend([
                            f"{op} {p.name} speed" for p in
                            PoseEstimationV3.KeypointIndex])
                    elif feature == 'closest_distances':
                        feature_list.append(f"{op} closest social distance")
                    elif feature == 'closest_fov_distances':
                        feature_list.append(f"{op} closest social distance in FoV")
                    elif feature == 'closest_fov_angles':
                        feature_list.append(f"{op} angle of closest social distance in FoV")
                    elif feature == 'social_pairwise_distances':
                        feature_list.extend([
                            f"{op} social dist. {sdn}"
                            for sdn in IdentityFeatures.get_social_distance_names()])
                    elif feature == 'social_pairwise_fov_distances':
                        feature_list.extend([
                            f"{op} social fov dist. {sdn}"
                            for sdn in IdentityFeatures.get_social_distance_names()])
                    else:
                        feature_list.extend(feature)

        return feature_list

    @classmethod
    def merge_per_frame_features(cls, features):
        """
        merge a list of per-frame features where each element in the list is
        a set of per-frame features computed for an individual animal
        :param features: list of per-frame feature instances
        :return: dict of the form
        {
            'pairwise_distances':,
            'angles':,
            'point_speeds':,
            'point_masks':
        }
        """

        # determine which features are in common
        feature_intersection = set(cls._per_frame_features + cls._per_frame_social_features)
        for feature_dict in features:
            feature_intersection &= set(feature_dict.keys())

        return {
            feature_name: np.concatenate([x[feature_name] for x in features])
            for feature_name in feature_intersection
        }

    @classmethod
    def merge_window_features(cls, features):
        """
        merge a list of window features where each element in the list is the
        set of window features computed for an individual animal
        :param features: list of window feature instances
        :return: dictionary of the form:
        {
            'angles' {
                'mean': numpy float32 array with shape (#frames, #angles),
                'median': numpy float32 array with shape (#frames, #angles),
                'std_dev': numpy float32 array with shape (#frames, #angles),
                'max': numpy float32 array with shape (#frames, #angles),
                'min' numpy float32 array with shape (#frames, #angles),
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
            'percent_frames_present': numpy float32 array with shape (#frames,)
        }
        """
        merged = {}

        # determine which features are in common
        feature_intersection = set(
            cls._window_features
            + cls._window_social_features)
        for feature_dict in features:
            feature_intersection &= set(feature_dict.keys())

        for feature_name in feature_intersection:
            if feature_name == 'percent_frames_present':
                merged[feature_name] = np.concatenate(
                    [x['percent_frames_present'] for x in features])
            else:
                merged[feature_name] = {}
                for op in cls._window_feature_operations:
                    merged[feature_name][op] = np.concatenate(
                        [x[feature_name][op] for x in features])

        return merged

    @staticmethod
    def _compute_pairwise_distance(points):
        """
        compute distances between all pairs of points
        :param points: collection of points
        :return: list of distances between all pairwise combinations of points
        """
        distances = []
        for i in range(0, len(points)):
            p1 = points[i]
            for p2 in points[i + 1:]:
                dist = math.hypot(int(p1[0]) - int(p2[0]), int(p1[1]) - int(p2[1]))
                distances.append(dist)
        return distances

    @staticmethod
    def _compute_social_pairwise_distance(points1, points2):
        """
        compute distances between all pairs of points
        :param points1: 1st collection of points
        :param points2: 2st collection of points
        :return: list of distances between all pairwise combinations of points
            from points1 and points2
        """
        distances = []

        for p1 in points1:
            for p2 in points2:
                dist = math.hypot(int(p1[0]) - int(p2[0]), int(p1[1]) - int(p2[1]))
                distances.append(dist)

        return distances

    @staticmethod
    def get_distance_names():
        """
        get list of human readable names for each value computed by
        _compute_pairwise_distances
        :return: list of distance names where each is a string of the form
        "distance_name_1-distance_name_2"
        """
        distances = []
        point_names = [p.name for p in PoseEstimationV3.KeypointIndex]
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

    @staticmethod
    def compute_angle(a, b, c):
        """
        compute angle created by three connected points
        :param a: point
        :param b: vertex point
        :param c: point
        :return: angle between AB and BC
        """
        angle = math.degrees(
            math.atan2(int(c[1]) - int(b[1]),
                       int(c[0]) - int(b[0])) - math.atan2(
                int(a[1]) - int(b[1]), int(a[0]) - int(b[0])))
        return angle + 360 if angle < 0 else angle

    @classmethod
    def _compute_angles(cls, keypoints):
        """
        compute angles between a subset of connected points.

        :param keypoints: 12 keypoints from pose estimation
        :return: numpy float array of computed angles, see AngleIndex enum for order
        """
        idx = PoseEstimationV3.KeypointIndex
        angles = np.empty(cls._num_angles, dtype=np.float32)

        angles[AngleIndex.NOSE_BASE_NECK_RIGHT_FRONT_PAW] = cls.compute_angle(
            keypoints[idx.NOSE],
            keypoints[idx.BASE_NECK],
            keypoints[idx.RIGHT_FRONT_PAW]
        )

        angles[AngleIndex.NOSE_BASE_NECK_LEFT_FRONT_PAW] = cls.compute_angle(
            keypoints[idx.NOSE],
            keypoints[idx.BASE_NECK],
            keypoints[idx.LEFT_FRONT_PAW]
        )

        angles[AngleIndex.RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE] = cls.compute_angle(
            keypoints[idx.RIGHT_FRONT_PAW],
            keypoints[idx.BASE_NECK],
            keypoints[idx.CENTER_SPINE]
        )

        angles[AngleIndex.LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE] = cls.compute_angle(
            keypoints[idx.LEFT_FRONT_PAW],
            keypoints[idx.BASE_NECK],
            keypoints[idx.CENTER_SPINE]
        )

        angles[AngleIndex.BASE_NECK_CENTER_SPINE_BASE_TAIL] = cls.compute_angle(
            keypoints[idx.BASE_NECK],
            keypoints[idx.CENTER_SPINE],
            keypoints[idx.BASE_TAIL]
        )

        angles[AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE] = cls.compute_angle(
            keypoints[idx.RIGHT_REAR_PAW],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.CENTER_SPINE]
        )

        angles[AngleIndex.LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE] = cls.compute_angle(
            keypoints[idx.LEFT_REAR_PAW],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.CENTER_SPINE]
        )

        angles[AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL] = cls.compute_angle(
            keypoints[idx.RIGHT_REAR_PAW],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.MID_TAIL]
        )

        angles[AngleIndex.LEFT_REAR_PAW_BASE_TAIL_MID_TAIL] = cls.compute_angle(
            keypoints[idx.LEFT_REAR_PAW],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.MID_TAIL]
        )

        angles[AngleIndex.CENTER_SPINE_BASE_TAIL_MID_TAIL] = cls.compute_angle(
            keypoints[idx.CENTER_SPINE],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.MID_TAIL]
        )

        angles[AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL] = cls.compute_angle(
            keypoints[idx.BASE_TAIL],
            keypoints[idx.MID_TAIL],
            keypoints[idx.TIP_TAIL]
        )

        return angles

    @staticmethod
    def _compute_point_speeds(poses, point_masks):
        """
        compute point speeds for post currently selected identity
        :param poses: pose estimations for an identity
        :param point_masks: corresponding point masks for poses
        :return: numpy array with shape (#frames, #key points)
        """

        num_frames = poses.shape[0]

        # generate an array of indexes so numpy gradient will know the spacing
        # between values since there may be gaps
        # should we convert to time based values rather than frame numbers?
        indexes = np.arange(num_frames)
        point_velocities = np.zeros(poses.shape)

        # calculate velocities for each point
        for point_index in range(poses.shape[1]):

            # grab all of the values for this point
            points = poses[:, point_index, :]

            # get the mask for each point too
            masks = point_masks[:, point_index]

            # and the indexes for the frames where the mask == 1
            valid_indexes = indexes[masks == 1]

            # if there are > 1 frame indexes where this point is valid, compute
            # the velocities
            if valid_indexes.shape[0] > 1:
                point_velocities[masks == 1, point_index, :] = np.gradient(
                    points[masks == 1], valid_indexes, axis=0)

        # convert the velocities to speed
        return np.linalg.norm(point_velocities, axis=-1)
