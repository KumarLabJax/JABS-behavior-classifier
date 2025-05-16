import numpy as np

import cv2

from jabs.pose_estimation import PoseEstimation
from jabs.feature_extraction.feature_base_class import Feature

_EXCLUDED_POINTS = [PoseEstimation.KeypointIndex.MID_TAIL,
                    PoseEstimation.KeypointIndex.TIP_TAIL]


class FoodHopper(Feature):
    _name = 'food_hopper'
    _min_pose = 5
    _static_objects = ['food_hopper']

    def per_frame(self, identity: int) -> dict:
        """get the per frame feature values for the food hopper landmark

        Args:
            identity: identity to get feature values for

        Returns:
            numpy ndarray of values with shape (nframes, 10)

        for each frame, the 10 feature values are the signed distance from the key point
        to the polygon defined by the food hopper key points (10 points
        because the mid tail and tail tip are excluded)
        """
        hopper = self._poses.static_objects['food_hopper']
        if self._pixel_scale is not None:
            hopper = hopper * self._pixel_scale

        # change dtype to float32 for open cv
        hopper_pts = hopper.astype(np.float32)

        points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        values = {}

        # for each keypoint (except mid tail and tail tip), compute signed distance (measureDist=True)
        # to the polygon defined by the food hopper key points. Distance is negative if the key point
        # is outside the polygon, positive if inside, zero if it is on the edge
        for key_point in PoseEstimation.KeypointIndex:
            # skip over the key points we don't care about
            if key_point in _EXCLUDED_POINTS:
                continue

            pts = points[:, key_point.value, :]

            distance = np.asarray([cv2.pointPolygonTest(hopper_pts, (p[0], p[1]), True) for p in pts])
            distance[np.isnan(pts[:, 0])] = np.nan
            values[f'food hopper {key_point.name}'] = distance

        return values

