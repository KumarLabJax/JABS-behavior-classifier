import typing

import cv2
import numpy as np
from shapely.geometry import Point

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation


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

    def cache_features(self, identity: int):
        """cache corner distances and bearings for a given identity

        Args:
            identity: integer identity to get distances for
        """
        if identity in self._cached_distances and identity in self._cached_bearings:
            return

        corner_distances = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        center_distances = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        wall_distances = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        all_wall_distances = np.full([self._poses.num_frames, 4], np.nan, dtype=np.float32)
        center_bearings = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        corner_bearings = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        closest_corners = None
        self_convex_hulls = self._poses.get_identity_convex_hulls(identity)
        idx = PoseEstimation.KeypointIndex
        avg_wall_length = np.nan

        if "corners" in self._poses.static_objects:
            closest_corners = np.full(self._poses.num_frames, -1, dtype=np.int8)
            corners = self.sort_points_clockwise(self._poses.static_objects["corners"])
            wall_vectors = corners.astype(np.float32) - np.roll(
                corners.astype(np.float32), 1, axis=0
            )
            avg_wall_length = (
                np.mean(np.hypot(wall_vectors[:, 0], wall_vectors[:, 1])) * self._pixel_scale
            )

            arena_center_np = np.mean(corners, axis=0)
            arena_center = Point(arena_center_np[0], arena_center_np[1])

            for frame in range(self._poses.num_frames):
                # don't scale the point coordinates by the pixel_scale value,
                # since the corners are in pixel space. we'll adjust the
                # distance units later
                points, mask = self._poses.get_points(frame, identity)

                if points is None:
                    continue

                # find distance to closest corner
                self_shape = self_convex_hulls[frame]
                distance = float("inf")
                corner_coordinates = (0, 0)
                closest_idx = -1
                for i in range(4):
                    d = self_shape.distance(Point(corners[i, 0], corners[i, 1]))
                    if d < distance:
                        distance = d
                        corner_coordinates = (corners[i, 0], corners[i, 1])
                        closest_idx = i

                self_base_neck_point = points[idx.BASE_NECK, :]
                self_nose_point = points[idx.NOSE, :]

                corner_bearing = self.compute_angle(
                    self_nose_point, self_base_neck_point, corner_coordinates
                )

                center_bearing = self.compute_angle(
                    self_nose_point, self_base_neck_point, arena_center_np
                )

                center_dist = self_shape.distance(arena_center)

                # Calculate distance to all walls using cross product
                p1 = corners.astype(np.float32)
                p2 = np.roll(p1, 1, axis=0)
                centroid_point = np.asarray(self_shape.centroid.xy).squeeze()

                # Ensure 3D vectors for np.cross to avoid deprecation warning
                centroid_point_3d = np.hstack([centroid_point, 0])
                p1_3d = np.hstack([p1, np.zeros((p1.shape[0], 1), dtype=np.float32)])
                p2_3d = np.hstack([p2, np.zeros((p2.shape[0], 1), dtype=np.float32)])

                # Note that we can skip dividing by the norm of p2-p1 because we re-scale it anyway
                wall_dist = np.abs(
                    np.cross(centroid_point_3d - p1_3d, p2_3d - p1_3d)
                )  # shape (N, 3)
                wall_dist = wall_dist[:, 2]  # Take the z-component

                shortest_wall_dist = np.min(wall_dist)
                wall_dist_cv2 = cv2.pointPolygonTest(
                    corners.astype(np.float32), centroid_point, True
                )
                correction_scale = wall_dist_cv2 / shortest_wall_dist
                wall_dist *= correction_scale
                wall_dist = np.abs(wall_dist)

                corner_distances[frame] = distance * self._pixel_scale
                center_distances[frame] = center_dist * self._pixel_scale
                wall_distances[frame] = wall_dist_cv2 * self._pixel_scale
                all_wall_distances[frame] = wall_dist * self._pixel_scale
                corner_bearings[frame] = corner_bearing
                center_bearings[frame] = center_bearing
                closest_corners[frame] = closest_idx

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

        Args:
            identity: identity to cache if not yet calculated

        Returns:
            average wall length
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

    @staticmethod
    def compute_angle(a, b, c):
        """compute angle created by three connected points

        Args:
            a: point
            b: vertex point
            c: point

        Returns:
            angle between AB and BC with range [-180, 180)
        """
        # most of the point types are unsigned short integers
        # cast to signed types to avoid underflow issues during subtraction
        angle = np.degrees(
            np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        )
        return ((angle + 180) % 360) - 180


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
