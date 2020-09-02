import enum
import math
import numpy as np
import h5py
from pathlib import Path

from src.pose_estimation import PoseEstimationV3
from src.labeler.track_labels import TrackLabels


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

    _num_distances = n_choose_r(len(PoseEstimationV3.KeypointIndex), 2)
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

    _per_frame_features = [
        'angles',
        'pairwise_distances',
        'point_speeds'
    ]

    _window_features = [
        'percent_frames_present',
        'angles',
        'pairwise_distances'
    ]

    def __init__(self, video_name, identity, directory, pose_est):
        """
        :param video_name: name of the video file, used for generating filenames
        for saving extracted features into the project directory
        :param identity: identity to extract features for
        :param directory: path of the project directory
        :param pose_est: PoseEstimationV3 object corresponding to this video
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

        # does this identity exist for this frame?
        self._frame_valid = np.zeros(self._num_frames, dtype=np.int8)

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

        for frame in range(pose_est.num_frames):
            points, confidence = pose_est.get_points(frame, self._identity)

            if points is not None:
                self._per_frame['pairwise_distances'][frame] = self._compute_pairwise_distance(points)
                self._per_frame['angles'][frame] = self._compute_angles(points)

        # indicate this identity exists in this frame
        self._frame_valid = pose_est.identity_mask(self._identity)

        self._per_frame['point_speeds'] = self._compute_point_speeds(pose_est)

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
            for op in features['pairwise_distances']:
                grp.create_dataset(f'pairwise_distances/{op}',
                                   data=features['pairwise_distances'][op])

            for op in features['angles']:
                grp.create_dataset(f'angles/{op}',
                                   data=features['angles'][op])

            grp.create_dataset('percent_frames_present',
                               data=features['percent_frames_present'])

    def _load_window_features(self, radius):
        """
        load window features from an h5 file
        :param radius: window size specified as the number of frames on each
        side of current frame to include in the window
        :return:
        """
        path = self._identity_feature_dir / f"window_features_{radius}.h5"

        window_features = {
            'pairwise_distances': {},
            'angles': {}
        }

        with h5py.File(path, 'r') as features_h5:
            # TODO verify file version
            assert features_h5.attrs['num_frames'] == self._num_frames
            assert features_h5.attrs['identity'] == self._identity
            assert features_h5.attrs['radius'] == radius

            feature_grp = features_h5['features']
            window_features['percent_frames_present'] = feature_grp['percent_frames_present'][:]

            for op in feature_grp['pairwise_distances'].keys():
                window_features['pairwise_distances'][op] = feature_grp[f'pairwise_distances/{op}'][:]
                assert len(window_features['pairwise_distances'][op]) == self._num_frames

            for op in feature_grp['angles'].keys():
                window_features['angles'][op] = feature_grp[f'angles/{op}'][:]
                assert len(window_features['angles'][op]) == self._num_frames

            return window_features

    def get_window_features(self, radius, labels=None):
        """
        get window features for a given window size, computing if not previously
        computed and saved as h5 file
        :param radius: number of frames on each side of the current frame to
        include in the window
        :param labels: optional frame labels, if present then only features for
        labeled frames will be returned
        :return: window features for given window size. the format is documented
        in the docstring for _compute_window_features
        """
        path = self._identity_feature_dir / f"window_features_{radius}.h5"

        if path.exists():
            # h5 file exists for this window size, load it
            features = self._load_window_features(radius)
        else:
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
                if key == 'radius':
                    filtered_features[key] = features[key]
                elif key == 'percent_frames_present':
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
                'pairwise_distances': self._per_frame['pairwise_distances'][labels != TrackLabels.Label.NONE, :],
                'angles': self._per_frame['angles'][labels != TrackLabels.Label.NONE, :],
                'point_speeds': self._per_frame['point_speeds'][labels != TrackLabels.Label.NONE, :]
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
        for feature in self._per_frame_features:
            per_frame[feature] = self._per_frame[feature][
                                 filter, :]

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
            'pairwise_distances': {}
        }

        # allocate arrays
        for operation in self._window_feature_operations:
            window_features['angles'][operation] = np.empty(
                [self._num_frames, self._num_angles], dtype=np.float32)

        for operation in self._window_feature_operations:
            window_features['pairwise_distances'][operation] = np.empty(
                [self._num_frames, self._num_distances], dtype=np.float32)

        window_features['percent_frames_present'] = np.empty((self._num_frames, 1),
                                                             dtype=np.float32)

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
                                angle_index][frame_valid]

                for operation in self._window_feature_operations:
                    window_features['angles'][operation][i, angle_index] = \
                        self._window_feature_operations[operation](window_values)

            # compute window features for distances
            for distance_index in range(0, self._num_distances):
                window_values = self._per_frame['pairwise_distances'][
                                    slice_start:slice_end,
                                    distance_index][frame_valid]

                for operation in self._window_feature_operations:
                    window_features['pairwise_distances'][operation][
                        i, distance_index] = self._window_feature_operations[
                        operation](window_values)

        return window_features

    @classmethod
    def get_feature_names(cls):
        """
        return list of human readable feature names, starting with per frame
        features followed by window features. feature names in each group are
        sorted
        :return: list of human readable feature names
        """
        feature_list = []
        for feature in sorted(cls._per_frame_features):
            if feature == 'angles':
                feature_list.extend([f"angle {angle.name}" for angle in AngleIndex])
            elif feature == 'pairwise_distances':
                feature_list.extend(IdentityFeatures.get_distance_names())
            elif feature == 'point_speeds':
                feature_list.extend([f"{point.name} speed" for point in PoseEstimationV3.KeypointIndex])

        for feature in sorted(cls._window_features):
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

        return feature_list

    @classmethod
    def merge_per_frame_features(cls, features):
        """
        merge a list of per-frame features where each element in the list is
        a set of per-frame features computed for an individual animal
        :param features: list of per-frame feature instances
        :return: dict of the form
        {
            'pairwise_distances': 2D numpy array,
            'angles': 2D numpy array
        }
        """

        return {
            'pairwise_distances': np.concatenate(
                [x['pairwise_distances'] for x in features]),
            'angles': np.concatenate([x['angles'] for x in features]),
            'point_speeds': np.concatenate([x['point_speeds'] for x in features])
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
            'percent_frames_present': numpy float32 array with shape (#frames,)
        }
        """
        merged = {
            'pairwise_distances': {},
            'angles': {},
            'percent_frames_present': np.concatenate(
                [x['percent_frames_present'] for x in features])
        }

        for op in cls._window_feature_operations:
            merged['pairwise_distances'][op] = np.concatenate(
                [x['pairwise_distances'][op] for x in features])
            merged['angles'][op] = np.concatenate(
                [x['angles'][op] for x in features])
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


    @staticmethod
    def _compute_angle(a, b, c):
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

        angles[AngleIndex.NOSE_BASE_NECK_RIGHT_FRONT_PAW] = cls._compute_angle(
            keypoints[idx.NOSE],
            keypoints[idx.BASE_NECK],
            keypoints[idx.RIGHT_FRONT_PAW]
        )

        angles[AngleIndex.NOSE_BASE_NECK_LEFT_FRONT_PAW] = cls._compute_angle(
            keypoints[idx.NOSE],
            keypoints[idx.BASE_NECK],
            keypoints[idx.LEFT_FRONT_PAW]
        )

        angles[AngleIndex.RIGHT_FRONT_PAW_BASE_NECK_CENTER_SPINE] = cls._compute_angle(
            keypoints[idx.RIGHT_FRONT_PAW],
            keypoints[idx.BASE_NECK],
            keypoints[idx.CENTER_SPINE]
        )

        angles[AngleIndex.LEFT_FRONT_PAW_BASE_NECK_CENTER_SPINE] = cls._compute_angle(
            keypoints[idx.LEFT_FRONT_PAW],
            keypoints[idx.BASE_NECK],
            keypoints[idx.CENTER_SPINE]
        )

        angles[AngleIndex.BASE_NECK_CENTER_SPINE_BASE_TAIL] = cls._compute_angle(
            keypoints[idx.BASE_NECK],
            keypoints[idx.CENTER_SPINE],
            keypoints[idx.BASE_TAIL]
        )

        angles[AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_CENTER_SPINE] = cls._compute_angle(
            keypoints[idx.RIGHT_REAR_PAW],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.CENTER_SPINE]
        )

        angles[AngleIndex.LEFT_REAR_PAW_BASE_TAIL_CENTER_SPINE] = cls._compute_angle(
            keypoints[idx.LEFT_REAR_PAW],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.CENTER_SPINE]
        )

        angles[AngleIndex.RIGHT_REAR_PAW_BASE_TAIL_MID_TAIL] = cls._compute_angle(
            keypoints[idx.RIGHT_REAR_PAW],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.MID_TAIL]
        )

        angles[AngleIndex.LEFT_REAR_PAW_BASE_TAIL_MID_TAIL] = cls._compute_angle(
            keypoints[idx.LEFT_REAR_PAW],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.MID_TAIL]
        )

        angles[AngleIndex.CENTER_SPINE_BASE_TAIL_MID_TAIL] = cls._compute_angle(
            keypoints[idx.CENTER_SPINE],
            keypoints[idx.BASE_TAIL],
            keypoints[idx.MID_TAIL]
        )

        angles[AngleIndex.BASE_TAIL_MID_TAIL_TIP_TAIL] = cls._compute_angle(
            keypoints[idx.BASE_TAIL],
            keypoints[idx.MID_TAIL],
            keypoints[idx.TIP_TAIL]
        )

        return angles

    def _compute_point_speeds(self, pose_est):
        """
        compute point speeds for post currently selected identity
        :param pose_est: pose estimations
        :return: numpy array with shape (#frames, #key points)
        """
        # get poses and point masks for current identity
        poses, point_masks = pose_est.get_identity_poses(self._identity)

        # generate an array of indexes so numpy gradient will no spaceing
        # between values since there may be gaps
        # TODO convert this to time based values rather than frame numbers
        indexes = np.arange(pose_est.num_frames)
        point_velocities = np.zeros(poses.shape)

        # calculate velocities for each point
        for point in pose_est.KeypointIndex:
            point_index = point.value

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


