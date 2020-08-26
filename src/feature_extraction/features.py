import enum
import math
import numpy as np
import h5py
from pathlib import Path
from itertools import compress

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

    def __init__(self, num_frames, identity, directory):

        self._num_frames = num_frames
        self._identity = identity
        self._project_feature_directory = Path(directory)

        # does this identity exist for this frame?
        self._frame_valid = np.zeros(num_frames, dtype=np.int8)

        # per frame features
        self._per_frame = {
            "pairwise_distances": np.empty([num_frames, self._num_distances],
                                           dtype=np.float32),

            "angles":  np.empty([num_frames, self._num_angles],
                                dtype=np.float32)
        }

    @classmethod
    def generate_from_pose_estimation(cls, pose_est, identity, directory):
        """
        Generate features from a PoseEstimation object, save_per_frame them in an h5
        file, and return an initialized IdentityFeatures object.

        :param pose_est: PoseEstimation object
        :param identity: identity to extract features for from pose_est
        :param directory: project feature directory
        :return: initialized IdentityFeatures object
        """

        features = cls(pose_est.num_frames, identity, directory)

        for frame in range(pose_est.num_frames):
            points, confidence = pose_est.get_points(frame, identity)

            if points is not None:
                features._per_frame["pairwise_distances"][frame] = cls._compute_pairwise_distance(points)
                features._per_frame["angles"][frame] = cls._compute_angles(points)

                # indicate this identity exists in this frame
                features._frame_valid[frame] = 1

        features.save_per_frame()
        return features

    @classmethod
    def load_from_file(cls, directory, identity):
        """
        return IdentityFeatures object initialized from state previously saved
        in a h5 file on disk
        :param directory: project feature directory
        :param identity: identity to load features for
        :return: initialized IdentityFeatures object
        """

        path = Path(directory) / str(identity) / 'per_frame.h5'

        with h5py.File(path, 'r') as features_h5:
            features = cls(features_h5.attrs['num_frames'],
                           features_h5.attrs['identity'],
                           directory)

            feature_grp = features_h5['features']

            # load per frame features
            features._frame_valid = features_h5['frame_valid'][:]
            features._per_frame["pairwise_distances"] = feature_grp['pairwise_distances'][:]
            features._per_frame["angles"] = feature_grp['angles'][:]

            assert len(features._frame_valid) == features._num_frames
            assert len(features._per_frame["pairwise_distances"]) == features._num_frames
            assert len(features._per_frame["angles"]) == features._num_frames

            return features

    def save_per_frame(self):
        """ save per frame features to a h5 file """

        identity_feature_dir = self._project_feature_directory / str(
            self._identity)
        identity_feature_dir.mkdir(mode=0o775, exist_ok=True)

        path = identity_feature_dir / 'per_frame.h5'

        with h5py.File(path, 'w') as features_h5:
            features_h5.attrs['num_frames'] = self._num_frames
            features_h5.attrs['identity'] = self._identity
            features_h5.attrs['version'] = self._version
            features_h5.create_dataset('frame_valid', data=self._frame_valid)

            grp = features_h5.create_group('features')
            grp.create_dataset('pairwise_distances', data=self._per_frame['pairwise_distances'])
            grp.create_dataset('angles', data=self._per_frame['angles'])

    def save_window_features(self, features):
        """
        save window features to an h5 file
        :param features: window features to save
        :return: None
        """
        path = (
            self._project_feature_directory /
            str(self._identity) /
            f"window_features_{features['radius']}.h5"
        )

        with h5py.File(path, 'w') as features_h5:
            features_h5.attrs['num_frames'] = self._num_frames
            features_h5.attrs['identity'] = self._identity
            features_h5.attrs['version'] = self._version

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
        path = (
                self._project_feature_directory /
                str(self._identity) /
                f"window_features_{radius}.h5"
        )

        window_features = {
            'radius': radius,
            'pairwise_distances': {},
            'angles': {}
        }

        with h5py.File(path, 'r') as features_h5:
            # TODO verify file version
            assert features_h5.attrs['num_frames'] == self._num_frames
            assert features_h5.attrs['identity'] == self._identity

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
        path = (
                self._project_feature_directory /
                str(self._identity) /
                f"window_features_{radius}.h5"
        )

        if path.exists():
            # h5 file exists for this window size, load it
            features = self._load_window_features(radius)
        else:
            # h5 file does not exist for this window size. compute the features
            # and return after saving
            features = self._compute_window_features(radius)
            self.save_window_features(features)

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
            'angles': 2D numpy array
        }
        """
        if labels is None:
            return self._per_frame

        else:
            # return only features for labeled frames
            return {
                "pairwise_distances": self._per_frame["pairwise_distances"][labels != TrackLabels.Label.NONE, :],
                "angles": self._per_frame["angles"][labels != TrackLabels.Label.NONE, :]
            }

    def _compute_window_features(self, radius):
        """
        compute all window features using a given window size
        :param radius: number of frames on each side of the current frame to
        include in the window
        :return: dictionary of the form:
        {
            'radius': int,
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
            'radius': radius,
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

        window_features['percent_frames_present'] = np.empty(self._num_frames,
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

            window_features['percent_frames_present'][
                i] = frames_in_window / max_window_size

            # compute window features for angles
            for angle_index in range(0, self._num_angles):
                window_values = self._per_frame['angles'][slice_start:slice_end,
                                angle_index]

                # filter window values
                filtered_slice = np.array(
                    list(compress(window_values, frame_valid)))

                for operation in self._window_feature_operations:
                    window_features['angles'][operation][i, angle_index] = \
                    self._window_feature_operations[operation](filtered_slice)

            # compute window features for distances
            for distance_index in range(0, self._num_distances):
                window_values = self._per_frame['pairwise_distances'][
                                slice_start:slice_end, distance_index]

                # filter window values
                filtered_slice = np.array(
                    list(compress(window_values, frame_valid)))

                for operation in self._window_feature_operations:
                    window_features['pairwise_distances'][operation][
                        i, distance_index] = self._window_feature_operations[
                        operation](filtered_slice)

        return window_features

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
