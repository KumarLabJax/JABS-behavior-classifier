import typing

import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature
from src.utils.utilities import smooth


class PointSpeeds(Feature):

    _name = 'point_speeds'
    _feature_names = [f"{p.name} speed" for p in PoseEstimation.KeypointIndex]

    def __init__(self, poses: PoseEstimation, pixel_scale: float):
        super().__init__(poses, pixel_scale)

    def per_frame(self, identity: int) -> np.ndarray:
        """
        compute the value of the per frame features for a specific identity
        :param identity: identity to compute features for
        :return: np.ndarray with feature values
        """
        num_frames = self._poses.num_frames
        fps = self._poses.fps
        poses, point_masks = self._poses.get_identity_poses(identity, self._pixel_scale)

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
                    points[masks == 1],
                    valid_indexes, axis=0)

        # convert the velocities to speed and convert units
        speeds = np.linalg.norm(point_velocities, axis=-1) * fps

        # smooth speeds
        for point_index in range(speeds.shape[1]):
            speeds[:, point_index] = smooth(speeds[:, point_index],
                                            smoothing_window=3)
        return speeds
