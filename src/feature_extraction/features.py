import enum
import math
from pathlib import Path

import h5py
import numpy as np
import scipy.stats

import src.project.track_labels
from src.utils.utilities import rolling_window, smooth
from src.pose_estimation import PoseEstimationV3

FEATURE_VERSION = 1


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


def get_angle_name(i: AngleIndex):
    strings = {
        AngleIndex.NOSE_BASE_NECK_RIGHT_FRONT_PAW: "NOSE - BASE_NECK - RIGHT_FRONT_PAW",
        AngleIndex.NOSE_BASE_NECK_LEFT_FRONT_PAW: "NOSE - BASE_NECK - LEFT_FRONT_PAW",
        AngleIndex.RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE: "RIGHT_FRONT_PAW - BASE_NECK - CENTER_SPINE",
        AngleIndex.LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE: "LEFT_FRONT_PAW - BASE_NECK - CENTER_SPINE",
        AngleIndex.BASE_NECK_CENTER_SPINE_BASE_TAIL: "BASE_NECK - CENTER_SPINE - BASE_TAIL",
        AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE: "RIGHT_REAR_PAW - BASE_TAIL - CENTER_SPINE",
        AngleIndex.LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE: "LEFT_REAR_PAW - BASE_TAIL - CENTER_SPINE",
        AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL: "RIGHT_REAR_PAW - BASE_TAIL - MID_TAIL",
        AngleIndex.LEFT_REAR_PAW_BASE_TAIL_MID_TAIL: "LEFT_REAR_PAW - BASE_TAIL - MID_TAIL",
        AngleIndex.CENTER_SPINE_BASE_TAIL_MID_TAIL: "CENTER_SPINE - BASE_TAIL - MID_TAIL",
        AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL: "BASE_TAIL - MID_TAILL - TIP_TAIL"
    }
    return strings[i]


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
        'velocity_mag',
        'velocity_dir'
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
        'velocity_mag',
        'velocity_dir'
    ]

    _window_social_features = [
        'closest_distances',
        'closest_fov_distances',
        'closest_fov_angles',
        'social_pairwise_distances',
        'social_pairwise_fov_distances',
    ]

    _circular_features = ['angles']
    _circular_features_2 = ['velocity_dir', 'closest_fov_angles']

    # TODO  For now this is taken from the ICY paper where the full field of
    # view is 240 degrees. Do we want this to be configurable?
    half_fov_deg = 120

    def __init__(self, video_name, identity, directory, pose_est, force=False,
                 fps=30):
        """
        :param video_name: name of the video file, used for generating filenames
        for saving extracted features into the project directory. You can use
        None for this argument if directory is also set to None
        :param identity: identity to extract features for
        :param directory: path of the project directory. A value of None can
        be given to prevent saving to and loading from a project dir.
        :param pose_est: PoseEstimationV3 object corresponding to this video
        :param force: force regeneration of per frame features even if the
        per frame feature .h5 file exists for this video/identity
        :param fps: frames per second. Used for converting angular velocity from
        degrees per frame to degrees per second
        """

        self._num_frames = pose_est.num_frames
        self._fps = fps
        self._identity = identity
        self._identity_feature_dir = None if directory is None else (
                Path(directory) /
                Path(video_name).stem /
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
                self._per_frame[feature] = np.zeros(
                    (self._num_frames, self._num_distances), dtype=np.float32)
            elif feature == 'angles':
                self._per_frame[feature] = np.zeros(
                    (self._num_frames, self._num_angles), dtype=np.float32)
            elif feature == 'point_speeds':
                self._per_frame[feature] = np.zeros(
                    (self._num_frames, len(PoseEstimationV3.KeypointIndex)),
                    dtype=np.float32)
            elif feature == 'point_mask':
                self._per_frame[feature] = \
                    pose_est.get_identity_point_mask(identity)
            elif feature in ['velocity_mag', 'velocity_dir']:
                self._per_frame[feature] = np.zeros(self._num_frames,
                                                    dtype=np.float32)
            elif feature == 'angular_velocity':
                continue
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

        if force or self._identity_feature_dir is None:
            self.__initialize_from_pose_estimation(pose_est)
        else:
            try:
                # try to load from an h5 file if it exists
                self.__load_from_file()
            except OSError:
                # otherwise compute the per frame features and save
                self.__initialize_from_pose_estimation(pose_est)

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
            *pose_est.get_identity_poses(self._identity), self._fps)

        bearings = pose_est.compute_all_bearings(self._identity)
        self._per_frame['angular_velocity'] = \
            smooth(self._compute_angular_velocities(bearings, self._fps),
                   smoothing_window=5)

        # compute the velocity of the center of mass.
        # first, grab convex hulls for this identity
        convex_hulls = pose_est.get_identity_convex_hulls(self._identity)

        # get an array of the indexes of valid frames only
        indexes = np.arange(self._num_frames)[self._frame_valid == 1]

        # get centroids for all frames where this identity is present
        centroids = [convex_hulls[i].centroid for i in indexes]

        # convert to numpy array of x,y points of the centroids
        points = np.asarray([[p.x, p.y] for p in centroids])

        # compute x,y velocities, pass indexes so numpy can figure out spacing
        v = np.gradient(points, indexes, axis=0)

        # compute magnitude and direction of velocities
        self._per_frame['velocity_mag'][indexes] = np.sqrt(np.square(v[:, 0]) + np.square(v[:, 1]))
        d = np.degrees(np.arctan2(v[:, 1], v[:, 0]))

        # subtract animal bearing from orientation
        # convert angle to range -180 to 180
        self._per_frame['velocity_dir'][indexes] = (((d - bearings[indexes]) + 360) % 360) - 180

        if self._identity_feature_dir is not None:
            self.save_per_frame()

    def __load_from_file(self):
        """
        initialize from state previously saved in a h5 file on disk
        This method will throw an exception if this object
        was constructed with a value of None for directory
        :return: None
        """

        path = self._identity_feature_dir / 'per_frame.h5'

        with h5py.File(path, 'r') as features_h5:
            feature_grp = features_h5['features']

            self._frame_valid = features_h5['frame_valid'][:]

            # load per frame features
            for feature in self._per_frame_features:
                if feature in ['point_mask']:
                    continue
                self._per_frame[feature] = feature_grp[feature][:]

            # TODO verify file version
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
            features_h5.create_dataset('frame_valid', data=self._frame_valid)

            grp = features_h5.create_group('features')
            grp.create_dataset('pairwise_distances', data=self._per_frame['pairwise_distances'])
            grp.create_dataset('angles', data=self._per_frame['angles'])
            grp.create_dataset('point_speeds', data=self._per_frame['point_speeds'])
            grp.create_dataset('angular_velocity',
                               data=self._per_frame['angular_velocity'])
            grp.create_dataset('velocity_mag',
                               data=self._per_frame['velocity_mag'])
            grp.create_dataset('velocity_dir',
                               data=self._per_frame['velocity_dir'])

            if self._include_social_features:
                features_h5['closest_identities'] = self._closest_identities
                features_h5['closest_fov_identities'] = self._closest_fov_identities

                for feature in self._per_frame_social_features:
                    grp[feature] = self._per_frame[feature]

    def save_window_features(self, features, window_size):
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
            features_h5.attrs['num_frames'] = self._num_frames
            features_h5.attrs['identity'] = self._identity
            features_h5.attrs['version'] = self._version
            features_h5.attrs['window_size'] = window_size

            grp = features_h5.create_group('features')

            for feature in self._window_features:
                for op in features[feature]:
                    grp.create_dataset(f'{feature}/{op}',
                                       data=features[feature][op])

            if self._include_social_features:
                for feature_name in self._window_social_features:
                    for op in features[feature_name]:
                        grp.create_dataset(f'{feature_name}/{op}',
                                           data=features[feature_name][op])

    def _load_window_features(self, window_size):
        """
        load window features from an h5 file
        :param window_size: window size specified as the number of frames on
        each side of current frame, in addition to the current frame, to
        include in the window
        (so if size=5, the total number of frames in the window is actually 11)
        :raises OSError: if unable to open h5 file
        :raises TypeError: if this object was constructed with a value of None
        for directory

        :return: window feature dict
        """
        path = self._identity_feature_dir / f"window_features_{window_size}.h5"

        window_features = {}

        with h5py.File(path, 'r') as features_h5:

            # early versions of the window feature h5 file called 'window_size'
            # 'radius', so fall back to that if 'window_size' isn't in the
            # attributes
            try:
                size_attr = features_h5.attrs['window_size']
            except KeyError:
                size_attr = features_h5.attrs['radius']

            # TODO verify file version
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

            if self._include_social_features:
                for feature_name in self._window_social_features:
                    window_features[feature_name] = {}
                    for op in feature_grp[feature_name].keys():
                        window_features[feature_name][op] = feature_grp[f'{feature_name}/{op}'][:]
                        assert len(window_features[feature_name][op]) == self._num_frames

            return window_features

    def get_window_features(self, window_size, labels=None, force=False):
        """
        get window features for a given window size, computing if not previously
        computed and saved as h5 file
        :param window_size: number of frames on each side of the current frame to
        include in the window
        :param labels: optional frame labels, if present then only features for
        labeled frames will be returned
        NOTE: if labels is None, this will include also values for frames where
        the identity does not exist. These get filtered out when filtering out
        unlabeled frames, since those frames are always unlabeled.
        :param force: force regeneration of the window features even if the
        h5 file already exists
        :return: window features for given window size. the format is documented
        in the docstring for _compute_window_features
        """

        if force or self._identity_feature_dir is None:
            features = self._compute_window_features(window_size)
            if self._identity_feature_dir is not None:
                self.save_window_features(features, window_size)

        else:
            try:
                # h5 file exists for this window size, load it
                features = self._load_window_features(window_size)
            except OSError:
                # h5 file does not exist for this window size. compute the
                # features and return after saving

                features = self._compute_window_features(window_size)

                if self._identity_feature_dir is not None:
                    self.save_window_features(features, window_size)

        if labels is None:
            return features
        else:
            # return only features for labeled frames
            filtered_features = {}

            for key in features:
                filtered_features[key] = {}
                for op in features[key]:
                    filtered_features[key][op] = features[key][op][labels != src.project.track_labels.TrackLabels.Label.NONE]

            return filtered_features

    def get_per_frame(self, labels=None):
        """
        get per frame features
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
            return self._per_frame

        else:
            # return only features for labeled frames
            return {
                k: v[labels != src.project.track_labels.TrackLabels.Label.NONE, ...]
                for k, v in self._per_frame.items()
            }

    def get_unlabeled_features(self, window_size, labels):
        """
        get features and corresponding frame indexes for unlabeled frames for
        classification
        :param window_size: number of frames before and after current frame to
        include in window feature calculation
        :param labels: array of labels (no label, not behavior, behavior)
        :return: dictionary with the following keys:

          'per_frame': dict with feature name as keys, numpy array as values
          'window': dict, see _compute_window_features
          'frame_indexes': 1D np array, maps elements of per frame and window
             feature arrays back to global frame indexes
        """
        window_features = self.get_window_features(window_size)
        filter = np.logical_and(self._frame_valid,
                                labels == src.project.track_labels.TrackLabels.Label.NONE)

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
            window[key] = {}
            for op in window_features[key]:
                window[key][op] = window_features[key][op][filter]

        return {
            'per_frame': per_frame,
            'window': window,
            'frame_indexes': indexes
        }

    def get_features(self, window_size):
        """
        get features and corresponding frame indexes for unlabeled frames for
        classification
        :param window_size:
        :return: dictionary with the following keys (see
        get_unlabeled_features for a description):
          'per_frame'
          'window'
          'frame_indexes'
        """
        window_features = self.get_window_features(window_size)

        per_frame = {}
        indexes = np.arange(self._num_frames)[self._frame_valid==1]

        if self._include_social_features:
            all_features = self._per_frame_features + self._per_frame_social_features
        else:
            all_features = self._per_frame_features

        for feature in all_features:
            per_frame[feature] = self._per_frame[feature][
                self._frame_valid==1, ...]

        window = {}
        for key in window_features:
            window[key] = {}
            for op in window_features[key]:
                window[key][op] = window_features[key][op][self._frame_valid==1]

        return {
            'per_frame': per_frame,
            'window': window,
            'frame_indexes': indexes
        }

    def _compute_window_features(self, window_size):
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
            'point_speeds': {...}
        }
        """

        max_window_size = 2 * window_size + 1

        window_features = {}
        for feature in self._window_features:
            window_features[feature] = {}

        # allocate arrays
        for operation in self._window_feature_operations_circular:
            window_features['angles'][operation] = np.zeros(
                [self._num_frames, self._num_angles], dtype=np.float32)
            window_features['velocity_dir'][operation] = np.zeros(
                self._num_frames, dtype=np.float32)

        for operation in self._window_feature_operations:
            window_features['pairwise_distances'][operation] = np.zeros(
                [self._num_frames, self._num_distances], dtype=np.float32)
            window_features['point_speeds'][operation] = np.zeros(
                [self._num_frames, len(PoseEstimationV3.KeypointIndex)],
                dtype=np.float32)
            window_features['angular_velocity'][operation] = np.zeros(
                self._num_frames, dtype=np.float32)
            window_features['velocity_mag'][operation] = np.zeros(
                self._num_frames, dtype=np.float32)

        # allocate arrays for social
        if self._include_social_features:
            for feature_name in self._window_social_features:
                window_features[feature_name] = {}
                if feature_name in ('social_pairwise_distances', 'social_pairwise_fov_distances'):
                    for operation in self._window_feature_operations:
                        window_features[feature_name][operation] = np.zeros(
                            (self._num_frames, self._num_social_distances),
                            dtype=np.float32)
                else:
                    if feature_name in self._circular_features:
                        operations = self._window_feature_operations_circular
                    elif feature_name in self._circular_features_2:
                        operations = self._window_feature_operations_circular_2
                    else:
                        operations = self._window_feature_operations
                    for operation in operations:
                        window_features[feature_name][operation] = np.zeros(
                            self._num_frames, dtype=np.float32)

        mask = np.full(self._num_frames, 1)
        mask[self._frame_valid == 1] = 0
        window_masks = rolling_window(
            np.pad(mask, window_size, 'constant', constant_values=(1)),
            max_window_size
        )

        if self._include_social_features:
            full_window_features = self._window_features + self._window_social_features
        else:
            full_window_features = self._window_features

        for op_name, op in self._window_feature_operations.items():
            for feature_name in full_window_features:
                if feature_name in self._circular_features + self._circular_features_2:
                    # these get handled elsewhere
                    continue

                if self._per_frame[feature_name].ndim == 1:
                    windows = rolling_window(
                        np.pad(self._per_frame[feature_name], window_size),
                        max_window_size
                    )
                    mx = np.ma.masked_array(windows, window_masks)
                    window_features[feature_name][op_name][:] = op(mx, axis=1)
                else:
                    for j in range(self._per_frame[feature_name].shape[1]):
                        windows = rolling_window(
                            np.pad(self._per_frame[feature_name][:, j], window_size),
                            max_window_size
                        )
                        mx = np.ma.masked_array(windows, window_masks)
                        window_features[feature_name][op_name][:, j] = op(mx, axis=1)

        # compute circular window features
        for i in range(self._num_frames):

            # identity doesn't exist for this frame don't bother to compute
            if not self._frame_valid[i]:
                continue

            slice_start = max(0, i - window_size)
            slice_end = min(i + window_size + 1, self._num_frames)

            frame_valid = self._frame_valid[slice_start:slice_end]

            # compute window features for angles
            for angle_index in range(0, self._num_angles):
                window_values = self._per_frame['angles'][slice_start:slice_end,
                                angle_index][frame_valid == 1]
                for op_name, op in self._window_feature_operations_circular.items():
                    if op_name == 'std_dev':
                        # XXX
                        # scipy.stats.circstd has a bug that can result in nan
                        # and a warning message to stderr if passed an array of
                        # nearly identical values
                        # this will be fixed when 1.6.0 is released, so this
                        # work-around can be removed once we can upgrade to
                        # scipy 1.6.0
                        # our work around is to suppress the warning and replace
                        # the nan with 0
                        with np.errstate(invalid='ignore'):
                            val = op(window_values)
                        if np.isnan(val):
                            window_features['angles'][op_name][i, angle_index] = 0.0
                        else:
                            window_features['angles'][op_name][i, angle_index] = val
                    else:
                        window_features['angles'][op_name][i, angle_index] = op(window_values)

            # compute window features for velocity_dir
            for op_name, op in self._window_feature_operations_circular_2.items():
                window_values = self._per_frame['velocity_dir'][slice_start:slice_end][frame_valid == 1]
                if op_name == 'std_dev':
                    # see comment for std_dev above
                    with np.errstate(invalid='ignore'):
                        val = op(window_values)
                    if np.isnan(val):
                        window_features['velocity_dir'][op_name][i] = 0.0
                    else:
                        window_features['velocity_dir'][op_name][i] = val
                else:
                    window_features['velocity_dir'][op_name][i] = op(window_values)

            # handle circular social features
            if self._include_social_features:
                for feature_name in self._window_social_features:
                    window_values = self._per_frame[feature_name][slice_start:slice_end, ...]
                    assert window_values.ndim == 1 or window_values.ndim == 2
                    if feature_name in self._circular_features:
                        operations = self._window_feature_operations_circular
                    elif feature_name in self._circular_features_2:
                        operations = self._window_feature_operations_circular_2
                    else:
                        continue
                    for op_name, op in operations.items():
                        if op_name == 'std_dev':
                            # XXX see comment above for explanation
                            with np.errstate(invalid='ignore'):
                                val = op(window_values[frame_valid == 1])
                            if np.isnan(val):
                                window_features[feature_name][
                                    op_name][i] = 0
                            else:
                                window_features[feature_name][
                                    op_name][i] = val
                        else:
                            window_features[feature_name][op_name][i] = op(
                                window_values[frame_valid == 1])
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
                    f"angle {get_angle_name(angle)}" for angle in AngleIndex])
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
            elif feature == 'velocity_mag':
                feature_list.append("animal velocity magnitude")
            elif feature == 'velocity_dir':
                feature_list.append("animal velocity orientation")
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
                            [f"{op} angle {get_angle_name(angle)}" for angle in AngleIndex])
                    elif feature == 'closest_fov_angles':
                        feature_list.append(f"{op} angle of closest social distance in FoV")
                    elif feature == 'velocity_dir':
                        feature_list.append(f"{op} velocity orientation")
            else:
                for op in sorted(cls._window_feature_operations):
                    if feature == 'pairwise_distances':
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
                    elif feature == 'social_pairwise_distances':
                        feature_list.extend([
                            f"{op} social dist. {sdn}"
                            for sdn in IdentityFeatures.get_social_distance_names()])
                    elif feature == 'social_pairwise_fov_distances':
                        feature_list.extend([
                            f"{op} social fov dist. {sdn}"
                            for sdn in IdentityFeatures.get_social_distance_names()])
                    elif feature == 'velocity_mag':
                        feature_list.append(f"{op} velocity magnitude")
                    else:
                        feature_list.append(f"{op} {feature}")

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

        # determine which features are in common
        feature_intersection = set(
            cls._window_features
            + cls._window_social_features)
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
    def _compute_point_speeds(poses, point_masks, fps):
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
        return np.linalg.norm(point_velocities, axis=-1) * fps

    @staticmethod
    def _compute_angular_velocities(angles, fps):
        velocities = np.zeros_like(angles)

        for i in range(len(angles) - 1):

            angle1 = angles[i]
            angle1 = angle1 % 360
            if angle1 < 0:
                angle1 += 360

            angle2 = angles[i + 1]
            angle2 = angle2 % 360
            if angle2 < 0:
                angle2 += 360

            diff1 = angle2 - angle1
            abs_diff1 = abs(diff1)
            diff2 = (360 + angle2) - angle1
            abs_diff2 = abs(diff2)
            diff3 = angle2 - (360 + angle1)
            abs_diff3 = abs(diff3)

            if abs_diff1 <= abs_diff2 and abs_diff1 <= abs_diff3:
                velocities[i] = diff1
            elif abs_diff2 <= abs_diff3:
                velocities[i] = diff2
            else:
                velocities[i] = diff3
        return velocities * fps
