import cv2
import numpy as np
import pytest

from jabs.feature_extraction.landmark_features.food_hopper import _EXCLUDED_POINTS
from jabs.pose_estimation import PoseEstimation


def test_dimensions(pose_est_v5_with_static_objects, food_hopper_feature):
    """Test dimensions of per frame and window feature values.

    Verifies that food hopper distance features have the correct shape
    for both per-frame values and window operation results.
    """
    for i in range(pose_est_v5_with_static_objects.num_identities):
        values = food_hopper_feature.per_frame(i)

        # TODO check dimensions of all key points, not just for NOSE
        assert values["food hopper NOSE"].shape == (pose_est_v5_with_static_objects.num_frames,)

        # check dimensions of window feature values
        dist_window_values = food_hopper_feature.window(i, 5, values)
        for op in dist_window_values:
            assert dist_window_values[op]["food hopper NOSE"].shape == (
                pose_est_v5_with_static_objects.num_frames,
            )


def test_signed_dist(pose_est_v5_with_static_objects, food_hopper_feature):
    """Test signed distance calculations match OpenCV reference implementation.

    Compares food hopper distance calculations against manual computations
    using cv2.pointPolygonTest to verify correctness.
    """
    values = food_hopper_feature.per_frame(0)

    # perform a couple manual computations of signed distance and check
    hopper = pose_est_v5_with_static_objects.static_objects["food_hopper"]
    if pose_est_v5_with_static_objects.cm_per_pixel is not None:
        hopper = hopper * pose_est_v5_with_static_objects.cm_per_pixel
    # swap the point x,y values and change dtype to float32 for open cv
    hopper_pts = hopper[:, [1, 0]].astype(np.float32)

    points, _ = pose_est_v5_with_static_objects.get_identity_poses(
        0, pose_est_v5_with_static_objects.cm_per_pixel
    )

    for key_point in PoseEstimation.KeypointIndex:
        # skip over the key points we don't care about
        if key_point in _EXCLUDED_POINTS:
            continue

        # swap our x,y to match the opencv coordinate space
        pts = points[:, key_point.value, [1, 0]]

        # check values for this keypoint for a few different frames
        for i in [5, 10, 50, 100, 200, 500, 1000]:
            signed_dist = cv2.pointPolygonTest(hopper_pts, (pts[i, 0], pts[i, 1]), True)
            if np.isnan(pts[i, 0]):
                signed_dist = np.nan

            if not np.isnan(signed_dist):
                assert signed_dist == pytest.approx(values[f"food hopper {key_point.name}"][i])
            else:
                assert np.isnan(values[f"food hopper {key_point.name}"][i])


def test_frame_out_of_range(food_hopper_feature):
    """Test that requesting a frame out of range raises IndexError."""
    with pytest.raises(IndexError):
        _ = food_hopper_feature.per_frame(0)["food hopper NOSE"][100000]


def test_identity_out_of_range(food_hopper_feature):
    """Test that requesting an identity out of range raises IndexError."""
    with pytest.raises(IndexError):
        _ = food_hopper_feature.per_frame(100)[0]
