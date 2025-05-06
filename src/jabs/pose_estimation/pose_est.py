import enum
import pickle
import typing
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np
from shapely.geometry import MultiPoint

from jabs.utils import hash_file

MINIMUM_CONFIDENCE = 0.3


class PoseHashException(Exception):
    pass


class PoseEstimation(ABC):
    """
    abstract base class for PoseEstimation objects. Used as the base class for
    PoseEstimationV2 and PoseEstimationV3
    """
    class KeypointIndex(enum.IntEnum):
        """ enum defining the 12 keypoint indexes """
        NOSE = 0
        LEFT_EAR = 1
        RIGHT_EAR = 2
        BASE_NECK = 3
        LEFT_FRONT_PAW = 4
        RIGHT_FRONT_PAW = 5
        CENTER_SPINE = 6
        LEFT_REAR_PAW = 7
        RIGHT_REAR_PAW = 8
        BASE_TAIL = 9
        MID_TAIL = 10
        TIP_TAIL = 11

    def __init__(self, file_path: Path, cache_dir: typing.Optional[Path] = None,
                 fps: int = 30):
        """
        initialize new object from h5 file
        :param file_path: path to pose_est_v2.h5 file
        :param cache_dir: optional cache directory, used to cache convex hulls
        for faster loading
        :param fps: frames per second, used for scaling time series features
        from "per frame" to "per second"
        """
        super().__init__()
        self._num_frames = 0
        self._identities = []
        self._convex_hull_cache = dict()
        self._path = file_path
        self._cache_dir = cache_dir
        self._cm_per_pixel = None
        self._hash = hash_file(file_path)
        self._fps = fps

        self._static_objects = {}

    @property
    def num_frames(self) -> int:
        """ return the number of frames in the pose_est file """
        return self._num_frames

    @property
    def identities(self):
        """ return list of integer identities generated from file """
        return self._identities

    @property
    def num_identities(self) -> int:
        return len(self._identities)

    @property
    def cm_per_pixel(self):
        return self._cm_per_pixel

    @property
    def fps(self):
        return self._fps

    @property
    def pose_file(self):
        return self._path

    @property
    def hash(self):
        return self._hash

    @abstractmethod
    def get_points(self, frame_index: int, identity: int,
                   scale: typing.Optional[float] = None):
        """
        return points and point masks for an individual frame
        :param frame_index: frame index of points and masks to be returned
        :param identity: identity to return points for
        :param scale: optional scale factor, set to cm_per_pixel to convert
        poses from pixel coordinates to cm coordinates
        :return: numpy array of points (12,2), numpy array of point masks (12,)
        """
        pass

    @abstractmethod
    def get_identity_poses(self, identity: int,
                           scale: typing.Optional[float] = None):
        """
        return all points and point masks
        :param identity: identity to return points for
        :param scale: optional scale factor, set to cm_per_pixel to convert
        poses from pixel coordinates to cm coordinates
        :return: numpy array of points (#frames, 12, 2), numpy array of point
        masks (#frames, 12)
        """
        pass

    @abstractmethod
    def get_identity_point_mask(self, identity):
        """
        get the point mask array for a given identity
        :param identity: identity to return point mask for
        :return: array of point masks (#frames, 12)
        """
        pass

    @abstractmethod
    def identity_mask(self, identity):
        """
        get the identity mask (indicates if specified identity is present in
        each frame)
        :param identity: identity to get masks for
        :return: numpy array of size (#frames,)
        """
        pass

    @property
    @abstractmethod
    def identity_to_track(self):
        pass

    @property
    @abstractmethod
    def format_major_version(self):
        """
        an integer giving the major version of the format
        """
        pass

    @property
    def static_objects(self):
        return self._static_objects

    def get_identity_convex_hulls(self, identity):
        """
        A list of length #frames containing convex hulls for the given identity.
        The convex hulls are calculated using all valid points except for the
        middle of tail and tip of tail points.
        :param identity: identity to return points for
        :return: the convex hulls in pixel units (array elements will be None
        if there is no valid convex hull for that frame)
        """

        if identity in self._convex_hull_cache:
            return self._convex_hull_cache[identity]
        else:
            convex_hulls = None
            path = None
            if self._cache_dir is not None:
                path = (self._cache_dir /
                        "convex_hulls" /
                        self._path.with_suffix('').name /
                        f"convex_hulls_{identity}.pickle")
                path.parents[0].mkdir(mode=0o775, parents=True, exist_ok=True)

                try:
                    with path.open('rb') as f:
                        convex_hulls = pickle.load(f)
                except:
                    # we weren't able to read in the cached convex hulls,
                    # just ignore the exception and we'll generate them
                    pass

            if convex_hulls is None:
                points, point_masks = self.get_identity_poses(identity)
                # Omit tail from convex hull
                body_points = points[:, :-2, :]
                body_point_masks = point_masks[:, :-2]
                convex_hulls = []

                for frame_index in range(self.num_frames):
                    if sum(body_point_masks[frame_index, :]) >= 3:
                        filtered_points = body_points[frame_index, body_point_masks[frame_index, :] == 1, :]
                        convex_hulls.append(MultiPoint(filtered_points).convex_hull)
                    else:
                        convex_hulls.append(None)

                if path:
                    with path.open('wb') as f:
                        pickle.dump(convex_hulls, f)

            self._convex_hull_cache[identity] = convex_hulls
            return convex_hulls

    def compute_bearing(self, points):
        base_tail_xy = points[self.KeypointIndex.BASE_TAIL.value].astype(np.float32)
        base_neck_xy = points[self.KeypointIndex.BASE_NECK.value].astype(np.float32)
        base_neck_offset_xy = base_neck_xy - base_tail_xy

        angle_rad = np.arctan2(base_neck_offset_xy[1],
                               base_neck_offset_xy[0])

        return np.degrees(angle_rad)

    def compute_all_bearings(self, identity):
        bearings = np.full(self.num_frames, np.nan, dtype=np.float32)
        for i in range(self.num_frames):
            points, mask = self.get_points(i, identity)
            if points is not None:
                bearings[i] = self.compute_bearing(points)
        return bearings

    @staticmethod
    def get_pose_file_attributes(path: Path) -> dict:
        with h5py.File(path, 'r') as pose_h5:
            attrs = dict(pose_h5.attrs)
            attrs['poseest'] = dict(pose_h5['poseest'].attrs)
            return attrs

    @property
    def lixit_keypoints(self) -> int:
        return 0
