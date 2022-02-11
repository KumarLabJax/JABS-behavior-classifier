import numpy as np
from shapely import geometry

from src.pose_estimation import PoseEstimation
from src.feature_extraction.feature_base_class import Feature

_EXCLUDED_POINTS = [PoseEstimation.KeypointIndex.MID_TAIL,
                    PoseEstimation.KeypointIndex.TIP_TAIL]


class FoodHopper(Feature):
    _name = 'food_hopper'
    _feature_names = [
        f'food hopper {p.name}'
        for p in PoseEstimation.KeypointIndex
        if p not in _EXCLUDED_POINTS
    ]
    _min_pose = 5
    _static_objects = ['food_hopper']

    def per_frame(self, identity: int) -> np.ndarray:
        """
        get the per frame feature values for the food hopper landmark
        :param identity: identity to get feature values for
        :return: numpy ndarray of values with shape (nframes, 10)
        for each frame, the 10 values indicated if the corresponding keypoint
        is on the food hopper (1) or not (0). (10 points because the mid tail
        and tail tip are excluded)
        """
        hopper = self._poses.static_objects['food_hopper']
        if self._pixel_scale is not None:
            hopper = hopper * self._pixel_scale

        hopper_poly = geometry.Polygon(hopper)
        points, _ = self._poses.get_identity_poses(identity, self._pixel_scale)

        values = np.zeros((self._poses.num_frames, len(self._feature_names)))

        for key_point in PoseEstimation.KeypointIndex:
            # skip over the few points we don't care about
            if key_point in _EXCLUDED_POINTS:
                continue

            # find out which frames the point is within the food hopper polygon
            hits = [hopper_poly.contains(geometry.Point(p)) for p in
                    points[:, key_point.value, :]]

            # set any frame where hits is true to 1
            values[hits, key_point.value] = 1.0

        return values

