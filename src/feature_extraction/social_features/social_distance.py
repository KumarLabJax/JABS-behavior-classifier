import math

import numpy as np

from src.pose_estimation import PoseEstimation


class ClosestIdentityInfo:
    """
    this info is needed to compute a number of different social features.
    It can be done once for a given identity, and then an instance of this
    object can be passed into all the features that need it
    """

    # TODO  For now this is taken from the ICY paper where the full field of
    # view is 240 degrees. Do we want this to be configurable?
    _half_fov_deg = 120

    def __init__(self, poses: PoseEstimation, identity: int,
                 pixel_scale: float):
        idx = PoseEstimation.KeypointIndex

        self._poses = poses
        self._identity = identity
        self._pixel_scale = pixel_scale

        self._closest_identities = np.full(poses.num_frames, -1,
                                           dtype=np.int16)
        self._closest_fov_identities = np.full(poses.num_frames, -1,
                                               dtype=np.int16)
        self._fov_angles = np.zeros(poses.num_frames, dtype=np.float32)

        for frame in range(poses.num_frames):
            points, mask = poses.get_points(frame, identity, pixel_scale)

            # skip this frame if the identity is not present
            if points is None:
                continue

            # Find the distance and identity of the closest animal at each
            # frame, as well as the distance, identity and angle of the closes
            # animal in field of view. In order to calculate this we require
            # that both animals have a valid convex hull and the the self
            # identity has a valid nose point and base neck point (which is
            # used to calculate FoV).
            self_shape = poses.get_identity_convex_hulls(identity)[frame]
            if self_shape is not None and mask[idx.NOSE] == 1 and mask[idx.BASE_NECK] == 1:
                closest_dist = None
                closest_fov_dist = None
                for curr_id in poses.identities:
                    if curr_id != identity:
                        other_shape = poses.get_identity_convex_hulls(curr_id)[frame]

                        if other_shape is not None:
                            curr_dist = self_shape.distance(other_shape)
                            if closest_dist is None or curr_dist < closest_dist:
                                self._closest_identities[frame] = curr_id
                                closest_dist = curr_dist

                            self_base_neck_point = points[idx.BASE_NECK, :]
                            self_nose_point = points[idx.NOSE, :]
                            other_centroid = np.array(other_shape.centroid.coords[0])

                            view_angle = self.compute_angle(
                                self_nose_point,
                                self_base_neck_point,
                                other_centroid)

                            # for FoV we want the range of view angle to be [180, -180)
                            if view_angle > 180:
                                view_angle -= 360

                            if abs(view_angle) <= self._half_fov_deg:
                                # other animal is in FoV
                                if closest_fov_dist is None or curr_dist < closest_fov_dist:
                                    self._closest_fov_identities[frame] = curr_id
                                    self._fov_angles[frame] = view_angle
                                    closest_fov_dist = curr_dist

    @property
    def closest_identities(self):
        return self._closest_identities

    @property
    def closest_fov_identities(self):
        return self._closest_fov_identities

    @property
    def closest_fov_angles(self):
        return self._fov_angles

    @staticmethod
    def compute_angle(a, b, c):
        """
        compute angle created by three connected points
        :param a: point
        :param b: vertex point
        :param c: point
        :return: angle between AB and BC
        """

        # point types in the pose files are typically unsigned 16 bit integers,
        # cast to signed types to avoid underflow during subtraction
        angle = math.degrees(
            math.atan2(int(c[1]) - int(b[1]), int(c[0]) - int(b[0])) -
            math.atan2(int(a[1]) - int(b[1]), int(a[0]) - int(b[0]))
        )
        return angle + 360 if angle < 0 else angle

    def compute_distances(self, closest_identities: np.ndarray) -> np.ndarray:
        values = np.zeros(self._poses.num_frames, dtype=np.float32)
        self_convex_hulls = self._poses.get_identity_convex_hulls(self._identity)

        for frame in range(self._poses.num_frames):

            points, mask = self._poses.get_points(frame, self._identity, self._pixel_scale)

            if points is None:
                continue

            closest_id = closest_identities[frame]
            if closest_id != -1:
                self_shape = self_convex_hulls[frame]
                other_shape = self._poses.get_identity_convex_hulls(closest_id)[
                    frame]
                values[frame] = self_shape.distance(other_shape)

        return values

    def compute_pairwise_social_distances(
            self, social_points: [PoseEstimation.KeypointIndex],
            closest_identities: np.ndarray
    ):
        values = np.zeros((self._poses.num_frames, len(social_points) ** 2),
                          dtype=np.float32)

        # get indexes of the subset of points used for pairwise social
        # distances
        social_pt_indexes = [idx.value for idx in social_points]

        for frame in range(self._poses.num_frames):
            points, mask = self._poses.get_points(frame, self._identity,
                                                  self._pixel_scale)

            if points is None or closest_identities[frame] == -1:
                continue

            closest_points, _ = self._poses.get_points(
                frame, closest_identities[frame], self._pixel_scale)

            values[frame] = self._compute_social_pairwise_distance(
                points[social_pt_indexes, ...],
                closest_points[social_pt_indexes, ...])
        return values

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
                dist = math.dist(p1, p2)
                distances.append(dist)
        return distances
