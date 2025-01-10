import cv2
import numpy as np

from src.jabs.pose_estimation import PoseEstimation
from src.jabs.feature_extraction.landmark_features.food_hopper import FoodHopper
from src.jabs.feature_extraction.landmark_features.food_hopper import _EXCLUDED_POINTS
from tests.feature_modules.base import TestFeatureBase


class TestFoodHopper(TestFeatureBase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        pixel_scale = cls._pose_est_v5.cm_per_pixel
        cls.food_hopper_feature = FoodHopper(cls._pose_est_v5, pixel_scale)

    def test_dimensions(self) -> None:
        # check dimensions of per frame feature values
        for i in range(self._pose_est_v5.num_identities):

            values = self.food_hopper_feature.per_frame(i)

            # TODO check dimensions of all key points, not just for NOSE
            self.assertEqual(values["food hopper NOSE"].shape, (self._pose_est_v5.num_frames,))

            # check dimensions of window feature values
            dist_window_values = self.food_hopper_feature.window(i, 5, values)
            for op in dist_window_values:
                self.assertEqual(dist_window_values[op]["food hopper NOSE"].shape, (self._pose_est_v5.num_frames,))

    def test_signed_dist(self) -> None:
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
                if np.isnan(pts[i, 0]):
                    signed_dist = np.nan

                if not np.isnan(signed_dist):
                    self.assertAlmostEqual(signed_dist, values[f"food hopper {key_point.name}"][i])
                else:
                    self.assertTrue(np.isnan(values[f"food hopper {key_point.name}"][i]))

    def test_frame_out_of_range(self) -> None:
        with self.assertRaises(IndexError):
            _ = self.food_hopper_feature.per_frame(0)["food hopper NOSE"][100000]

    def test_identity_out_of_range(self) -> None:
        with self.assertRaises(IndexError):
            _ = self.food_hopper_feature.per_frame(100)[0]
