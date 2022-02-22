import cv2
import numpy as np

from src.pose_estimation import PoseEstimation
from src.feature_extraction.landmark_features.food_hopper import FoodHopper
from src.feature_extraction.landmark_features.food_hopper import _EXCLUDED_POINTS
from tests.feature_modules.base import TestFeatureBase


class TestCornerFeatures(TestFeatureBase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        pixel_scale = cls._pose_est_v5.cm_per_pixel
        cls.food_hopper_feature = FoodHopper(cls._pose_est_v5, pixel_scale)

    def test_dimensions(self):
        # check dimensions of per frame feature values
        for i in range(self._pose_est_v5.num_identities):
            values = self.food_hopper_feature.per_frame(i)
            self.assertEqual(values.shape,
                             (self._pose_est_v5.num_frames,
                              len(self.food_hopper_feature.feature_names())))

            # check dimensions of window feature values
            dist_window_values = self.food_hopper_feature.window(i, 5, values)
            for op in dist_window_values:
                self.assertEqual(dist_window_values[op].shape,
                                 (self._pose_est_v5.num_frames,
                                  len(self.food_hopper_feature.feature_names())))

    def test_signed_dist(self):
        values = self.food_hopper_feature.per_frame(0)

        # perform a couple manual computations of signed distance and check
        hopper = self._pose_est_v5.static_objects['food_hopper']
        if self._pose_est_v5.cm_per_pixel is not None:
            hopper = hopper * self._pose_est_v5.cm_per_pixel
        # swap the point x,y values and change dtype to float32 for open cv
        hopper_pts = hopper[:, [1, 0]].astype(np.float32)

        points, _ = self._pose_est_v5.get_identity_poses(
            0, self._pose_est_v5.cm_per_pixel)

        for key_point in PoseEstimation.KeypointIndex:
            # skip over the key points we don't care about
            if key_point in _EXCLUDED_POINTS:
                continue

            # swap our x,y to match the opencv coordinate space
            pts = points[:, key_point.value, [1, 0]]

            # check values for this keypoint for a few different frames
            for i in [5, 10, 50, 100, 200, 500, 1000]:
                signed_dist = cv2.pointPolygonTest(
                    hopper_pts, (pts[i, 0], pts[i, 1]), True)
                self.assertAlmostEqual(signed_dist, values[i, key_point])
