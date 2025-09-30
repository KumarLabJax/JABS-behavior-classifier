import enum
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import joblib
import numpy as np
from shapely.geometry import MultiPoint

from jabs.utils import hash_file

MINIMUM_CONFIDENCE = 0.3


class PoseHashException(Exception):
    """Exception raised when the hash of a pose file does not match the expected value."""

    pass


class PoseEstimation(ABC):
    """Abstract base class for pose estimation data handlers.

    Provides a common interface for loading, accessing, and processing pose data
    from HDF5 files. Defines methods for retrieving keypoints, confidence masks, identity
    presence, and static objects, as well as utilities for geometric computations such as
    convex hulls and bearing angles. All pose estimation versioned classes should inherit
    from this base class.

    Args:
        file_path (Path): Path to the pose HDF5 file.
        cache_dir (Path | None): Optional cache directory for intermediate data.
        fps (int): Frames per second for the video.

    Abstract Methods:
        get_points(frame_index, identity, scale): Get points and mask for an identity in a frame.
        get_identity_poses(identity, scale): Get all points and masks for an identity.
        get_identity_point_mask(identity): Get the point mask array for a given identity.
        identity_mask(identity): Get the identity mask for a given identity.
        identity_to_track: Get the identity-to-track mapping for this file.
        format_major_version: Returns the major version of the pose file format.

    Methods:
        get_identity_convex_hulls(identity): Get convex hulls for an identity across frames.
        compute_bearing(points): Compute the bearing angle for a single frame.
        compute_all_bearings(identity): Compute bearing angles for all frames of an identity.
        get_pose_file_attributes(path): Static method to get HDF5 file attributes.

    Properties:
        num_frames (int): Number of frames.
        identities (list): List of identities.
        num_identities (int): Number of identities.
        cm_per_pixel (float | None): Centimeters per pixel.
        fps (int): Frames per second.
        pose_file (Path): Path to the pose file.
        hash (str): Hash of the pose file.
        static_objects (dict): Static objects in the pose file.
        num_lixit_keypoints (int): Number of lixit keypoints (default 0).
        external_identities (list[int] | None): Mapping to external identities.
    """

    class KeypointIndex(enum.IntEnum):
        """enum defining the 12 keypoint indexes"""

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

    # Connected segments to use when full 12 keypoints are available.
    FULL_CONNECTED_SEGMENTS = (
        (
            KeypointIndex.LEFT_FRONT_PAW,
            KeypointIndex.CENTER_SPINE,
            KeypointIndex.RIGHT_FRONT_PAW,
        ),
        (
            KeypointIndex.LEFT_REAR_PAW,
            KeypointIndex.BASE_TAIL,
            KeypointIndex.RIGHT_REAR_PAW,
        ),
        (
            KeypointIndex.NOSE,
            KeypointIndex.BASE_NECK,
            KeypointIndex.CENTER_SPINE,
            KeypointIndex.BASE_TAIL,
            KeypointIndex.MID_TAIL,
            KeypointIndex.TIP_TAIL,
        ),
    )

    # Pose based on the Envision Hydra model will have fewer keypoints,
    # so we adjust the connected segments accordingly.
    NVSN_CONNECTED_SEGMENTS = (
        (
            KeypointIndex.LEFT_EAR,
            KeypointIndex.NOSE,
            KeypointIndex.RIGHT_EAR,
        ),
        (
            KeypointIndex.NOSE,
            KeypointIndex.BASE_TAIL,
            KeypointIndex.TIP_TAIL,
        ),
    )

    _CACHE_FILE_VERSION = 1

    def __init__(self, file_path: Path, cache_dir: Path | None = None, fps: int = 30):
        """initialize new object from h5 file

        Args:
            file_path: path to pose_est_v2.h5 file
            cache_dir: optional cache directory, used to cache convex
                hulls
            fps: frames per second, used for scaling time series
                features
        for faster loading
        from "per frame" to "per second"
        """
        super().__init__()
        self._num_frames = 0
        self._identities = []
        self._external_identities: list[str] | None = None
        self._convex_hull_cache = {}
        self._path = file_path
        self._cache_dir = cache_dir
        self._cm_per_pixel = None
        self._hash = hash_file(file_path)
        self._fps = fps

        self._static_objects = {}

        # check cache version, if it doesn't match, clear the cache file for this pose file
        if self._cache_dir is not None and not self.check_cache_version():
            cache_file = self._cache_file_path()
            if cache_file and cache_file.exists():
                try:
                    cache_file.unlink()
                except Exception:
                    logging.warning("Unable to delete old cache file %s", cache_file)
                    pass

    @property
    def num_frames(self) -> int:
        """return the number of frames in the pose_est file"""
        return self._num_frames

    @property
    def identities(self):
        """return list of integer identities generated from file"""
        return self._identities

    @property
    def num_identities(self) -> int:
        """get the number of identities in the pose file"""
        return len(self._identities)

    @property
    def cm_per_pixel(self):
        """get centimeters per pixel for video/pose"""
        return self._cm_per_pixel

    @property
    def fps(self):
        """get frames per second"""
        return self._fps

    @property
    def pose_file(self):
        """get the path to the pose file"""
        return self._path

    @property
    def hash(self):
        """get the hash of the pose file"""
        return self._hash

    @abstractmethod
    def get_points(self, frame_index: int, identity: int, scale: float | None = None):
        """return points and point masks for an individual frame

        Args:
            frame_index: frame index of points and masks to be returned
            identity: identity to return points for
            scale: optional scale factor, set to cm_per_pixel to convert
                poses from pixel coordinates to cm coordinates

        Returns:
            numpy array of points (12,2), numpy array of point masks (12,)
        """
        pass

    @abstractmethod
    def get_identity_poses(self, identity: int, scale: float | None = None):
        """return all points and point masks

        Args:
            identity: identity to return points for
            scale: optional scale factor, set to cm_per_pixel to convert
                poses from pixel coordinates to cm coordinates

        Returns:
            numpy array of points (#frames, 12, 2), numpy array of point masks (#frames, 12)
        """
        pass

    @abstractmethod
    def get_identity_point_mask(self, identity):
        """get the point mask array for a given identity

        Args:
            identity: identity to return point mask for

        Returns:
            array of point masks (#frames, 12)
        """
        pass

    @abstractmethod
    def get_reduced_point_mask(self):
        """Returns a boolean array of length 12 indicating which keypoints are valid.

        Determines which keypoints are valid for any identity across all frames.

        Returns:
            numpy array of shape (12,) with boolean values indicating validity
            of each keypoint.
        """
        pass

    def get_connected_segments(self):
        """Get the segments to use for rendering connections between the keypoints

        Returns:
            list of tuples, where each tuple contains the indexes of the keypoints
            that form a connected segment
        """
        return PoseEstimation.FULL_CONNECTED_SEGMENTS

    @abstractmethod
    def identity_mask(self, identity):
        """get the identity mask (indicates if specified identity is present in each frame)

        Args:
            identity: identity to get masks for

        Returns:
            numpy array of size (#frames,)
        """
        pass

    @property
    @abstractmethod
    def identity_to_track(self):
        """get the identity to track mapping for this file"""
        pass

    @property
    @abstractmethod
    def format_major_version(self):
        """an integer giving the major version of the format"""
        pass

    @property
    def static_objects(self):
        """get static objects from the pose file"""
        return self._static_objects

    def get_identity_convex_hulls(self, identity):
        """get a list of length #frames containing convex hulls for the given identity.

        The convex hulls are calculated using all valid points except for the
        middle of tail and tip of tail points.

        Args:
            identity: identity to return points for

        Returns:
            the convex hulls in pixel units (array elements will be None
            if there is no valid convex hull for that frame)
        """
        if identity in self._convex_hull_cache:
            return self._convex_hull_cache[identity]
        else:
            convex_hulls = None
            path = None
            if self._cache_dir is not None:
                path = (
                    self._cache_dir
                    / "convex_hulls"
                    / self._path.with_suffix("").name
                    / f"convex_hulls_{identity}.pickle"
                )
                path.parents[0].mkdir(mode=0o775, parents=True, exist_ok=True)

                try:
                    with path.open("rb") as f:
                        convex_hulls = joblib.load(f)
                except Exception:
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
                        filtered_points = body_points[
                            frame_index, body_point_masks[frame_index, :] == 1, :
                        ]
                        convex_hulls.append(MultiPoint(filtered_points).convex_hull)
                    else:
                        convex_hulls.append(None)

                if path:
                    with path.open("wb") as f:
                        joblib.dump(convex_hulls, f)

            self._convex_hull_cache[identity] = convex_hulls
            return convex_hulls

    def compute_bearing(self, points):
        """compute the bearing of the animal using base tail and base neck keypoints

        Args:
            points (np.ndarray): the points for a single frame (12,2) array
        """
        base_tail_xy = points[self.KeypointIndex.BASE_TAIL.value].astype(np.float32)
        base_neck_xy = points[self.KeypointIndex.BASE_NECK.value].astype(np.float32)
        base_neck_offset_xy = base_neck_xy - base_tail_xy

        angle_rad = np.arctan2(base_neck_offset_xy[1], base_neck_offset_xy[0])

        return np.degrees(angle_rad)

    def compute_all_bearings(self, identity):
        """compute the bearing for each frame for a given identity"""
        bearings = np.full(self.num_frames, np.nan, dtype=np.float32)
        for i in range(self.num_frames):
            points, mask = self.get_points(i, identity)
            if points is not None:
                bearings[i] = self.compute_bearing(points)
        return bearings

    @staticmethod
    def get_pose_file_attributes(path: Path) -> dict:
        """get the attributes from the pose file's hdf5 file"""
        with h5py.File(path, "r") as pose_h5:
            attrs = dict(pose_h5.attrs)
            attrs["poseest"] = dict(pose_h5["poseest"].attrs)
            return attrs

    @property
    def num_lixit_keypoints(self) -> int:
        """get the number of lixit keypoints

        always 0 for pose file versions <5
        """
        return 0

    @property
    def external_identities(self) -> list[str] | None:
        """get the jabs identity to external identity mapping"""
        return self._external_identities

    def identity_index_to_display(self, identity_index: int) -> str:
        """Convert an identity index to a display string.

        Args:
            identity_index (int): The identity index to convert.

        Returns:
            str: The display string for the identity.
        """
        if self.external_identities and 0 <= identity_index < len(self.external_identities):
            return self.external_identities[identity_index]
        return str(identity_index)

    def check_cache_version(self) -> bool:
        """Check if the cache version matches the expected version.

        Returns:
            bool: True if the cache version matches, False otherwise.
        """
        try:
            with h5py.File(self._cache_file_path(), "r") as cache_h5:
                cache_version = cache_h5.attrs.get("cache_file_version", None)
                return cache_version == self._CACHE_FILE_VERSION
        except Exception:
            return False

    def _cache_file_path(self) -> Path | None:
        """Get the path to the cache file for this pose file.

        Returns:
            Path | None: The path to the cache file, or None if no cache directory is set.
        """
        if self._cache_dir is None:
            return None
        filename = self._path.name.replace(".h5", "_cache.h5")
        return self._cache_dir / filename
