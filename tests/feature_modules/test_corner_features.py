import numpy as np

import src.jabs.feature_extraction.landmark_features.corner as corner_module
from tests.feature_modules.base import TestFeatureBase


class TestCornerFeatures(TestFeatureBase):
    def test_compute_corner_distances(self):
        pixel_scale = self._pose_est_v5.cm_per_pixel
        dist = corner_module.CornerDistanceInfo(self._pose_est_v5, pixel_scale)
        dist_to_corner = corner_module.DistanceToCorner(
            self._pose_est_v5, pixel_scale, dist
        )
        bearing_to_corner = corner_module.BearingToCorner(
            self._pose_est_v5, pixel_scale, dist
        )

        # check dimensions of per frame feature values
        for i in range(self._pose_est_v5.num_identities):
            dist_per_frame = dist_to_corner.per_frame(i)

            self.assertEqual(
                dist_per_frame["distance to corner"].shape,
                (self._pose_est_v5.num_frames,),
            )

            bearing_per_frame = bearing_to_corner.per_frame(i)
            self.assertEqual(
                bearing_per_frame["bearing to corner"].shape,
                (self._pose_est_v5.num_frames,),
            )

            # check dimensions of window feature values
            dist_window_values = dist_to_corner.window(i, 5, dist_per_frame)
            for op in dist_window_values:
                self.assertEqual(
                    dist_window_values[op]["distance to corner"].shape,
                    (self._pose_est_v5.num_frames,),
                )

            bearing_window_values = bearing_to_corner.window(i, 5, bearing_per_frame)
            for op in bearing_window_values:
                self.assertEqual(
                    bearing_window_values[op]["bearing to corner"].shape,
                    (self._pose_est_v5.num_frames,),
                )

        # check range of bearings, should be in the range [180, -180)
        for i in range(self._pose_est_v5.num_identities):
            values = bearing_to_corner.per_frame(i)["bearing to corner"]
            non_nan_indices = ~np.isnan(values)
            self.assertTrue(
                (
                    (values[non_nan_indices] <= 180) & (values[non_nan_indices] > -180)
                ).all()
            )

        # check distances are >= 0
        for i in range(self._pose_est_v5.num_identities):
            values = dist_to_corner.per_frame(i)["distance to corner"]
            non_nan_indices = ~np.isnan(values)
            self.assertTrue((values[non_nan_indices] >= 0).all())
