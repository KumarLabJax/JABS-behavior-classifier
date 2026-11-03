import typing

import cv2
import numpy as np
import numpy.typing as npt
import shapely

from jabs.core.utils import signed_angle_degrees
from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation


def _scale_like_python_float(
    values_px: npt.NDArray[np.float64], pixel_scale: float
) -> npt.NDArray[np.floating]:
    """Scale pixel distances that were previously Python floats, one frame at a time.

    ``python_float * pixel_scale`` takes the precision of ``pixel_scale``: float32
    arithmetic for ``np.float32`` (as read from pose files), float64 for a Python float
    or ``np.float64``. This reproduces that so vectorized values are unchanged.

    Args:
        values_px: Distances in pixels.
        pixel_scale: Scale factor to convert pixel distances to cm.

    Returns:
        Scaled distances.
    """
    # same type as the original ``python_float * pixel_scale``; it is a float even when
    # pixel_scale is an integer
    return values_px.astype(np.result_type(1.0 * pixel_scale)) * pixel_scale


class CornerDistanceInfo:
    """cached distance to corner

    because we have two features that both need to know which corner is the
    closest, we compute that information once in this helper class and then
    pass it to both of the classes that implement the features.
    The features are not merged into a single feature class because one (bearing to
    corner) needs to use the circular window feature methods, and the other
    does not, so it can't easily be implemented as one multi-column Feature
    class.
    """

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        self._poses = poses
        self._pixel_scale = pixel_scale
        self._closest_corner_idx = {}
        self._cached_distances = {}
        self._cached_bearings = {}
        self._all_wall_distances = {}

        # None until cache_features() runs; it is then set to the arena's average
        # wall length, or NaN when the pose file has no "corners" static object.
        # get_avg_wall_length() relies on this None to know it must still compute.
        self._avg_wall_length: float | None = None

    def cache_features(self, identity: int):
        """cache corner distances and bearings for a given identity

        Args:
            identity: integer identity to get distances for
        """
        if identity in self._cached_distances and identity in self._cached_bearings:
            return

        num_frames = self._poses.num_frames
        corner_distances = np.full(num_frames, np.nan, dtype=np.float32)
        center_distances = np.full(num_frames, np.nan, dtype=np.float32)
        wall_distances = np.full(num_frames, np.nan, dtype=np.float32)
        all_wall_distances = np.full([num_frames, 4], np.nan, dtype=np.float32)
        center_bearings = np.full(num_frames, np.nan, dtype=np.float32)
        corner_bearings = np.full(num_frames, np.nan, dtype=np.float32)
        closest_corners = None
        avg_wall_length = np.nan

        if "corners" in self._poses.static_objects:
            closest_corners = np.full(num_frames, -1, dtype=np.int8)
            corners = self.sort_points_clockwise(self._poses.static_objects["corners"])
            corners_f32 = corners.astype(np.float32)
            wall_vectors = corners_f32 - np.roll(corners_f32, 1, axis=0)
            avg_wall_length = (
                np.mean(np.hypot(wall_vectors[:, 0], wall_vectors[:, 1])) * self._pixel_scale
            )
            arena_center = np.mean(corners, axis=0)

            # frames where the identity is present and has a convex hull. Hulls omit the
            # tail, so a present identity can occasionally lack one; those frames are
            # left as NaN like frames where the identity is absent.
            hulls = np.array(self._poses.get_identity_convex_hulls(identity), dtype=object)
            present = np.asarray(self._poses.identity_mask(identity), dtype=bool)
            frames = np.flatnonzero(present & ~shapely.is_missing(hulls))
            frame_hulls = hulls[frames]

            # don't scale the point coordinates by the pixel_scale value, since the
            # corners are in pixel space. we'll adjust the distance units later
            points, _ = self._poses.get_identity_poses(identity)
            idx = PoseEstimation.KeypointIndex
            nose = points[frames, idx.NOSE, :]
            base_neck = points[frames, idx.BASE_NECK, :]

            # distance to closest corner; argmin keeps the first corner on ties
            corner_dists = shapely.distance(
                frame_hulls[:, np.newaxis], shapely.points(corners)[np.newaxis, :]
            )
            closest_idx = np.argmin(corner_dists, axis=1)
            distance = corner_dists[np.arange(len(frames)), closest_idx]

            corner_bearing = signed_angle_degrees(nose, base_neck, corners[closest_idx])
            center_bearing = signed_angle_degrees(nose, base_neck, arena_center)
            center_dist = shapely.distance(frame_hulls, shapely.points(arena_center))

            centroids = shapely.get_coordinates(shapely.centroid(frame_hulls))
            all_walls, wall_dist_cv2 = self._wall_distances(centroids, corners_f32)

            corner_distances[frames] = _scale_like_python_float(distance, self._pixel_scale)
            center_distances[frames] = _scale_like_python_float(center_dist, self._pixel_scale)
            wall_distances[frames] = _scale_like_python_float(wall_dist_cv2, self._pixel_scale)
            all_wall_distances[frames] = all_walls * self._pixel_scale
            corner_bearings[frames] = corner_bearing
            center_bearings[frames] = center_bearing
            closest_corners[frames] = closest_idx

        self._cached_distances[identity] = {
            "distance to corner": corner_distances,
            "distance to center": center_distances,
            "distance to wall": wall_distances,
        }

        self._cached_bearings[identity] = {
            "bearing to corner": corner_bearings,
            "bearing to center": center_bearings,
        }

        self._closest_corner_idx[identity] = closest_corners
        self._all_wall_distances[identity] = {
            f"wall_{i}": all_wall_distances[:, i] for i in np.arange(all_wall_distances.shape[1])
        }
        self._avg_wall_length = avg_wall_length

    @staticmethod
    def _wall_distances(
        centroids: npt.NDArray[np.float64], corners: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute distances from hull centroids to each arena wall.

        Args:
            centroids: Hull centroids, shape (n, 2), in pixels.
            corners: Arena corners sorted clockwise, shape (4, 2), in pixels.

        Returns:
            Tuple of (distance to each wall, shape (n, 4); signed distance to the
            arena boundary from cv2.pointPolygonTest, positive inside, shape (n,)).
        """
        p1 = corners
        wall_vectors = np.roll(p1, 1, axis=0) - p1
        offsets = centroids[:, np.newaxis, :] - p1[np.newaxis, :, :]
        # z component of the cross product of (centroid - p1) and (p2 - p1). We skip
        # dividing by the norm of p2 - p1 because the values are re-scaled below.
        wall_dist = np.abs(
            offsets[..., 0] * wall_vectors[np.newaxis, :, 1]
            - offsets[..., 1] * wall_vectors[np.newaxis, :, 0]
        )
        wall_dist_cv2 = np.array(
            [cv2.pointPolygonTest(corners, (float(x), float(y)), True) for x, y in centroids],
            dtype=np.float64,
        )
        correction_scale = wall_dist_cv2 / np.min(wall_dist, axis=1)
        return np.abs(wall_dist * correction_scale[:, np.newaxis]), wall_dist_cv2

    def get_distances(self, identity: int) -> dict:
        """get corner distance features for a given identity

        Args:
            identity: integer identity to get distances for

        Returns:
            dict containing keyed distances
        """
        if identity not in self._cached_distances:
            self.cache_features(identity)
        return self._cached_distances[identity]

    def get_bearings(self, identity: int) -> dict:
        """get corner bearing features for a given identity

        Args:
            identity: integer identity to get bearings for

        Returns:
            dict containing keyed bearings
        """
        if identity not in self._cached_bearings:
            self.cache_features(identity)
        return self._cached_bearings[identity]

    def get_closest_corner(self, identity: int) -> np.ndarray:
        """get the closest corner index

        Args:
            identity: integer identity to get the closest corner

        Returns:
            np.ndarray of the corner index
        """
        if identity not in self._closest_corner_idx:
            self.cache_features(identity)
        return self._closest_corner_idx[identity]

    def get_wall_distances(self, identity: int) -> np.ndarray:
        """get the wall distances

        Args:
            identity: integer identity to get the wall distances

        Returns:
            np.ndarray of all wall distances
        """
        if identity not in self._all_wall_distances:
            self.cache_features(identity)
        return self._all_wall_distances[identity]

    def get_avg_wall_length(self, identity: int = 0) -> float:
        """gets the average wall length

        The average wall length is a property of the arena rather than of an
        individual mouse, so a single instance-wide value is shared by every
        identity. ``cache_features()`` derives it from the same static corners
        every time it runs for a new identity, overwriting the stored value
        with an identical one.

        Args:
            identity: identity to cache if not yet calculated

        Returns:
            average wall length, or NaN if the pose file has no corners
        """
        if self._avg_wall_length is None:
            self.cache_features(identity)
        return self._avg_wall_length

    @staticmethod
    def sort_points_clockwise(points):
        """sorts a list of points to be clockwise relative to the first point

        Args:
            points: points to sort of shape [n_points, 2]

        Returns:
            points sorted clockwise
        """
        origin_point = np.mean(points, axis=0)
        vectors = points - origin_point
        vec_angles = np.arctan2(vectors[:, 0], vectors[:, 1])
        sorted_points = points[np.argsort(vec_angles), :]
        # Roll the points to have the first point still be first
        first_point_idx = np.where(np.all(sorted_points == points[0], axis=1))[0][0]
        return np.roll(sorted_points, -first_point_idx, axis=0)


class DistanceToCorner(Feature):
    """Feature extraction class for computing the distance to the nearest corner.

    Args:
        poses (PoseEstimation): Pose estimation data for a video.
        pixel_scale (float): Scale factor to convert pixel distances to cm.
        distances (CornerDistanceInfo): Object providing corner distance information.
    """

    _name = "corner_distances"
    _min_pose = 5
    _static_objects: typing.ClassVar[list[str]] = ["corners"]

    def __init__(self, poses: PoseEstimation, pixel_scale: float, distances: CornerDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    def per_frame(self, identity: int) -> dict:
        """get the per frame distance to the nearest corner values

        Args:
            identity: identity to get feature values for

        Returns:
            dict of numpy ndarray of values with shape (nframes,)
        """
        distances = self._cached_distances.get_distances(identity)
        return distances


class BearingToCorner(Feature):
    """Feature extraction class for computing the bearing to the nearest corner.

    Args:
        poses (PoseEstimation): Pose estimation data for a video.
        pixel_scale (float): Scale factor to convert pixel distances to cm.
        distances (CornerDistanceInfo): Object providing corner distance and bearing information.
    """

    _name = "corner_bearings"
    _min_pose = 5
    _static_objects: typing.ClassVar[list[str]] = ["corners"]
    _use_circular = True

    def __init__(self, poses: PoseEstimation, pixel_scale: float, distances: CornerDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    def per_frame(self, identity: int) -> dict:
        """get the per frame bearing to the nearest corner values

        Args:
            identity: identity to get feature values for

        Returns:
            dict of numpy ndarray values with shape (nframes,)
        """
        features = self._cached_distances.get_bearings(identity)
        features["bearing to corner cosine"] = np.cos(np.deg2rad(features["bearing to corner"]))
        features["bearing to corner sine"] = np.sin(np.deg2rad(features["bearing to corner"]))
        features["bearing to center cosine"] = np.cos(np.deg2rad(features["bearing to center"]))
        features["bearing to center sine"] = np.sin(np.deg2rad(features["bearing to center"]))
        return features
