import typing

import numpy as np

from jabs.feature_extraction.feature_base_class import Feature
from jabs.pose_estimation import PoseEstimation


class LixitDistanceInfo:
    """compute and cache distances and bearings to lixit objects for reuse by multiple features.

    because we have two feature groups that both need to know which lixit is the closest, we compute that
    information once in this helper class and then pass it to both of the features. The features cannot be merged into
    a single feature because one requires a different set of window feature methods.
    """

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        # Identify closest lixit
        self._closest_key = PoseEstimation.KeypointIndex.NOSE
        # Distances include all keypoints
        self._keypoint_indices = list(PoseEstimation.KeypointIndex)

        self._poses = poses
        self._pixel_scale = pixel_scale
        self._closest_lixit_idx = {}
        self._cached_distances = {}
        self._cached_bearings = {}

    def cache_features(self, identity: int):
        """Computes and caches distances and bearings to the closest lixit for a given identity.

        Args:
            identity (int): The identity index for which to compute and cache lixit distances and bearings.

        Returns:
            None
        """
        # if we have already computed the distances and bearings for this identity, we don't need to do anything
        if identity in self._cached_distances and identity in self._cached_bearings:
            return

        self._closest_lixit_idx[identity] = None
        self._cached_distances[identity] = {
            f"distance to lixit {keypoint.name}": np.full(
                self._poses.num_frames, np.nan, dtype=np.float32
            )
            for keypoint in self._keypoint_indices
        }
        self._cached_bearings[identity] = {
            "bearing to lixit": np.full(self._poses.num_frames, np.nan, dtype=np.float32)
        }

        if "lixit" in self._poses.static_objects:
            lixit = self._poses.static_objects["lixit"]

            # there might be more than one: shape of lixit is <number of lixit> x 2
            # OR in newer versions, it might be <number of lixit> x 3 x 2 since a lixit can be defined by three points
            # (tip, left side, right side -- in that order) instead of just a single point (tip)
            num_lixit = lixit.shape[0]
            points_per_lixit = 3 if lixit.ndim == 3 else 1

            # convert lixit coordinates from pixels to cm if we can
            if self._pixel_scale is not None:
                lixit = lixit * self._pixel_scale

            alignment_distances = np.full(
                (self._poses.num_frames, num_lixit), np.nan, dtype=np.float32
            )

            points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

            # if there are multiple lixit, we compute the distance from nose to
            # each one, resulting in a numpy array of shape #frames, #lixit
            pts = points[:, self._closest_key, :]
            for i in range(num_lixit):
                # use the tip of the lixit to determine closest
                ref = lixit[i, 0] if points_per_lixit == 3 else lixit[i]
                alignment_distances[:, i] = np.sqrt(np.sum((pts - ref) ** 2, axis=1))

            self._closest_lixit_idx[identity] = np.argmin(alignment_distances, axis=1)

            if points_per_lixit == 3:
                # grab just the tip keypoint for determining pairwise distances from pose keypoints to lixit tip
                closest_lixit_vec = lixit[self._closest_lixit_idx[identity], 0]
            else:
                closest_lixit_vec = lixit[self._closest_lixit_idx[identity]]

            for keypoint in self._keypoint_indices:
                pts = points[:, keypoint, :]
                dists = pts - closest_lixit_vec
                kpt_dist_vector = np.hypot(dists[:, 0], dists[:, 1])
                self._cached_distances[identity][f"distance to lixit {keypoint.name}"] = (
                    kpt_dist_vector
                )

            nose_points = points[:, PoseEstimation.KeypointIndex.NOSE, :]
            base_neck_points = points[:, PoseEstimation.KeypointIndex.BASE_NECK, :]
            self._cached_bearings[identity]["bearing to lixit"] = self.compute_angles(
                nose_points, base_neck_points, closest_lixit_vec
            )

    def get_distances(self, identity: int) -> dict:
        """get lixit distance feature for a given identity

        Args:
            identity: integer identity to get distances for

        Returns:
            dict containing keyed distances
        """
        if identity not in self._cached_distances:
            self.cache_features(identity)
        return self._cached_distances[identity]

    def get_bearings(self, identity: int) -> dict:
        """get lixit bearing features for a given identity

        Args:
            identity: integer identity to get bearings for

        Returns:
            dict containing keyed bearings
        """
        if identity not in self._cached_bearings:
            self.cache_features(identity)
        return self._cached_bearings[identity]

    def get_closest_lixit(self, identity: int) -> np.ndarray:
        """get the closest lixit index

        Args:
            identity: integer identity to get the closest lixit

        Returns:
            np.ndarray of the lixit index
        """
        if identity not in self._closest_lixit_idx:
            self.cache_features(identity)
        return self._closest_lixit_idx[identity]

    @staticmethod
    def compute_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """compute angles for a set of points

        Args:
            a: array of point coordinates
            b: array of vertex point coordinates
            c: array of point coordinates

        Returns:
            array containing angles, in degrees, formed from the lines ab and ba for each row in a, b, and c with range [-180, 180)
        """
        angles = np.degrees(
            np.arctan2(c[:, 1] - b[:, 1], c[:, 0] - b[:, 0])
            - np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0])
        )
        return ((angles + 180) % 360) - 180


class DistanceToLixit(Feature):
    """
    Helper class to compute and cache distances and bearings from animal keypoints to lixit objects.

    This class determines, for each frame and identity, the closest lixit (water spout) and calculates:
      - The distance from each keypoint to the nearest lixit tip.
      - The bearing (angle) to the lixit using a vector from the animal's nose to base neck as the reference.
    Results are cached for efficient reuse by multiple feature extraction classes.

    Args:
        poses (PoseEstimation): Pose estimation data containing keypoints and static objects.
        pixel_scale (float): Scale factor to convert pixel distances to centimeters.
    """

    _name = "lixit_distances"
    _min_pose = 5
    _static_objects: typing.ClassVar[list[str]] = ["lixit"]

    def __init__(self, poses: PoseEstimation, pixel_scale: float, distances: LixitDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    def per_frame(self, identity: int) -> dict:
        """get the per frame distance to nearest lixit

        Args:
            identity: identity to get feature values for

        Returns:
            dict of numpy ndarray of values with shape (nframes,)
        """
        distances = self._cached_distances.get_distances(identity)
        return distances


class BearingToLixit(Feature):
    """This class computes the bearing to the nearest lixit for each frame.

    Args:
        poses (PoseEstimation): Pose estimation data for a video.
        pixel_scale (float): Scale factor to convert pixel distances to cm.
        distances (LixitDistanceInfo): Object providing pre-computed lixit distance and bearing information.
    """

    _name = "lixit_bearings"
    _min_pose = 5
    _static_objects: typing.ClassVar[list[str]] = ["lixit"]
    _use_circular = True

    def __init__(self, poses: PoseEstimation, pixel_scale: float, distances: LixitDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """get the per frame bearing to the nearest lixit values

        Args:
            identity: identity to get the feature values for

        Returns:
            dict of numpy ndarray values with shape (nframes,)
        """
        features = self._cached_distances.get_bearings(identity)
        features["bearing to lixit sine"] = np.sin(np.deg2rad(features["bearing to lixit"]))
        features["bearing to lixit cosine"] = np.cos(np.deg2rad(features["bearing to lixit"]))

        return features


class MouseLixitAngle(Feature):
    """This class computes two features.

    1. the angle between the vector going from the tip of the lixit to the middle of the two sides
    and the vector going from the centroid to the nose

    2. the angle between the vector going from the tip of the lixit to the middle of the two sides
    and the vector going from the base tail to the centroid

    The intention of including this feature is to provide a measure of how the mouse is oriented with respect to
    the lixit. Unlike the "bearing to lixit", which is a measure of the angle between the direction the mouse
    is facing and the lixit, this feature will allow the classifier to learn when the mouse is approaching the lixit
    from the front.

    Note: feature is actually computed as the cosine of the angle, which is more useful for classification
    """

    _name = "mouse_lixit_angle"
    _min_pose = 5
    _static_objects: typing.ClassVar[list[str]] = ["lixit"]

    def __init__(self, poses: PoseEstimation, pixel_scale: float, distances: LixitDistanceInfo):
        super().__init__(poses, pixel_scale)

        self._cached_distances = distances

    @classmethod
    def is_supported(cls, pose_version: int, static_objects: set[str], **kwargs) -> bool:
        """Check if the feature is supported based on the pose version and static objects."""
        return bool(
            super().is_supported(pose_version, static_objects, **kwargs)
            and kwargs.get("lixit_keypoints") == 3
        )

    def per_frame(self, identity: int) -> dict[str, np.ndarray]:
        """Calculate per frame features.

        Args:
            identity (int): The identity of the mouse for which the feature is being calculated.

        Returns:
            dict[str, np.ndarray]: A dictionary containing the cosine of the angle for each frame,
            clipped to the range [-1, 1].
        """
        # Get the poses (keypoint coordinates) for the given identity, scaled by pixel_scale
        points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        # get centroids
        # first, get an array of the indexes of valid frames only
        frame_valid = self._poses.identity_mask(identity)
        indexes = np.arange(self._poses.num_frames)[frame_valid == 1]

        # then get centroids for all frames where this identity is present
        convex_hulls = self._poses.get_identity_convex_hulls(identity)
        centroids = np.full([self._poses.num_frames, 2], np.nan, dtype=np.float32)
        for i in indexes:
            centroids[i, :] = np.asarray(convex_hulls[i].centroid.xy).squeeze()

        # Compute the vector from the nose to the center of the spine for the mouse
        mouse_vectors = points[:, PoseEstimation.KeypointIndex.NOSE, :] - centroids

        mouse_back_vectors = centroids - points[:, PoseEstimation.KeypointIndex.BASE_TAIL, :]

        # Get the lixit points for the closest lixit to the given identity
        lixit_points = self._poses.static_objects["lixit"][
            self._cached_distances.get_closest_lixit(identity)
        ]

        # Compute the vector from the tip of the lixit to the midpoint of its left and right sides
        lixit_vectors = lixit_points[:, 0, :] - (lixit_points[:, 1, :] + lixit_points[:, 2, :]) / 2

        # Compute the dot product of the mouse vectors and the lixit vectors
        dot_product = np.einsum("ij,ij->i", mouse_vectors, lixit_vectors)
        dot_product_back = np.einsum("ij,ij->i", mouse_back_vectors, lixit_vectors)

        # Compute the norms (magnitudes) of the mouse vectors and lixit vectors
        norm_mouse_vectors = np.linalg.norm(mouse_vectors, axis=1)
        norm_mouse_back_vectors = np.linalg.norm(mouse_back_vectors, axis=1)
        norm_lixit_vectors = np.linalg.norm(lixit_vectors, axis=1)

        # Compute the cosine of the angle between the two vectors and clip it to the range [-1, 1]
        return {
            "centroid - nose": np.clip(
                dot_product / (norm_mouse_vectors * norm_lixit_vectors), -1.0, 1.0
            ),
            "base-tail - centroid": np.clip(
                dot_product_back / (norm_mouse_back_vectors * norm_lixit_vectors),
                -1.0,
                1.0,
            ),
        }
