import math

import numpy as np

from jabs.pose_estimation import PoseEstimation


class ClosestIdentityInfo:
    """this info is needed to compute a number of different social features.

    It can be done once for a given identity, and then an instance of this object can be passed into all the
    features that need it
    """

    # TODO  For now this is taken from the ICY paper where the full field of
    # view is 240 degrees. Do we want this to be configurable?
    HALF_FOV_DEGREE = 120

    def __init__(self, poses: PoseEstimation, identity: int, pixel_scale: float):
        idx = PoseEstimation.KeypointIndex

        self._poses = poses
        self._identity = identity
        self._pixel_scale = pixel_scale

        self._closest_identities = np.full(poses.num_frames, -1, dtype=np.int16)
        self._closest_fov_identities = np.full(poses.num_frames, -1, dtype=np.int16)
        self._fov_angles = np.full(poses.num_frames, np.nan, dtype=np.float32)

        for frame in range(poses.num_frames):
            points, mask = poses.get_points(frame, identity, pixel_scale)

            # skip this frame if the identity is not present
            if points is None:
                continue

            # Find the distance and identity of the closest animal at each
            # frame, as well as the distance, identity and angle of the closest
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
                            other_centroid = (
                                np.array(other_shape.centroid.xy).squeeze() * self._pixel_scale
                            )

                            view_angle = self.compute_angle(
                                self_nose_point, self_base_neck_point, other_centroid
                            )

                            if abs(view_angle) <= self.HALF_FOV_DEGREE and (
                                closest_fov_dist is None or curr_dist < closest_fov_dist
                            ):
                                self._closest_fov_identities[frame] = curr_id
                                self._fov_angles[frame] = view_angle
                                closest_fov_dist = curr_dist

    @property
    def closest_identities(self):
        """Returns the closest identities for each frame."""
        return self._closest_identities

    @property
    def closest_fov_identities(self):
        """Returns the closest identities in the field of view for each frame."""
        return self._closest_fov_identities

    @property
    def closest_fov_angles(self):
        """Returns the angles of the closest animals in the field of view."""
        return self._fov_angles

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
        angle = np.degrees(
            np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        )
        return ((angle + 180) % 360) - 180

    def compute_distances(self, closest_identities: np.ndarray) -> np.ndarray:
        """Computes the frame-wise distances between the subject's convex hull and the convex hull of the closest other animal.

        Args:
            closest_identities (np.ndarray): Array of closest identity indices for each frame.

        Returns:
            np.ndarray: Array of distances for each frame, with NaN for frames where distance could not be computed.

        Todo:
         - we already compute distances in the constructor, we should save them and return them here
        """
        values = np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        self_convex_hulls = self._poses.get_identity_convex_hulls(self._identity)

        for frame in range(self._poses.num_frames):
            points, mask = self._poses.get_points(frame, self._identity, self._pixel_scale)

            if points is None:
                continue

            closest_id = closest_identities[frame]
            if closest_id != -1:
                self_shape = self_convex_hulls[frame]
                other_shape = self._poses.get_identity_convex_hulls(closest_id)[frame]
                values[frame] = self_shape.distance(other_shape)

        return values

    def compute_pairwise_social_distances(
        self,
        social_points: list[PoseEstimation.KeypointIndex],
        closest_identities: np.ndarray,
    ):
        """
        Computes pairwise distances between specified keypoints of the subject and the closest other identity.

        Args:
            social_points (list[PoseEstimation.KeypointIndex]): List of keypoint indices to use for pairwise distance
             calculation.
            closest_identities (np.ndarray): Array of closest identity indices for each frame.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping keypoint pair names to arrays of distances for each frame.
        """
        return_dict = {}
        for p1 in social_points:
            for p2 in social_points:
                return_dict[f"social dist. {p1.name}-{p2.name}"] = np.full(
                    self._poses.num_frames, np.nan, dtype=np.float32
                )

        for frame in range(self._poses.num_frames):
            closest_id = int(closest_identities[frame])
            points, mask = self._poses.get_points(frame, self._identity, self._pixel_scale)

            if points is None or closest_id == -1:
                continue

            closest_points, _ = self._poses.get_points(frame, closest_id, self._pixel_scale)

            for p1 in social_points:
                for p2 in social_points:
                    key = f"social dist. {p1.name}-{p2.name}"
                    return_dict[key][frame] = math.dist(points[p1.value], closest_points[p2.value])

        return return_dict
