from src.feature_extraction.landmark_features.food_hopper import FoodHopper
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

    def test_values(self):
        """tests that the food hopper point mask values are 0 or 1"""
        for i in range(self._pose_est_v5.num_identities):
            vals = self.food_hopper_feature.per_frame(i)
            # check distances are >= 0
            self.assertTrue(((vals == 0.0) | (vals == 1.0)).all())
