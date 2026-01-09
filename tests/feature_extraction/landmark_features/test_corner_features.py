import numpy as np

import jabs.feature_extraction.landmark_features.corner as corner_module


def test_compute_corner_distances(pose_est_v5_with_static_objects):
    """Test the computation of corner distances and bearings."""
    pose_est_v5 = pose_est_v5_with_static_objects
    pixel_scale = pose_est_v5.cm_per_pixel
    dist = corner_module.CornerDistanceInfo(pose_est_v5, pixel_scale)
    dist_to_corner = corner_module.DistanceToCorner(pose_est_v5, pixel_scale, dist)
    bearing_to_corner = corner_module.BearingToCorner(pose_est_v5, pixel_scale, dist)

    # check dimensions of per frame feature values
    for i in range(pose_est_v5.num_identities):
        dist_per_frame = dist_to_corner.per_frame(i)

        assert dist_per_frame["distance to corner"].shape == (pose_est_v5.num_frames,)

        bearing_per_frame = bearing_to_corner.per_frame(i)
        assert bearing_per_frame["bearing to corner"].shape == (pose_est_v5.num_frames,)

        # check dimensions of window feature values
        dist_window_values = dist_to_corner.window(i, 5, dist_per_frame)
        for op in dist_window_values:
            assert dist_window_values[op]["distance to corner"].shape == (pose_est_v5.num_frames,)

        bearing_window_values = bearing_to_corner.window(i, 5, bearing_per_frame)
        for op in bearing_window_values:
            for feature in bearing_window_values[op]:
                assert bearing_window_values[op][feature].shape == (pose_est_v5.num_frames,)

    # check range of bearings, should be in the range [180, -180)
    for i in range(pose_est_v5.num_identities):
        values = bearing_to_corner.per_frame(i)["bearing to corner"]
        non_nan_indices = ~np.isnan(values)
        assert ((values[non_nan_indices] <= 180) & (values[non_nan_indices] > -180)).all()

    # check distances are >= 0
    for i in range(pose_est_v5.num_identities):
        values = dist_to_corner.per_frame(i)["distance to corner"]
        non_nan_indices = ~np.isnan(values)
        assert (values[non_nan_indices] >= 0).all()
