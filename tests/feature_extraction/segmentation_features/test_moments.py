import numpy as np

from jabs.feature_extraction.segmentation_features import Moments


def test_posev6_get_segmentation_data_selects_identity(seg_data):
    """Test that PoseEstimationV6.get_segmentation_data returns each identity's contours.

    Verifies that the data returned for an identity is that identity's slice of the
    segmentation array, for every identity the file stores.
    """
    pose_est_v6 = seg_data["pose_est_v6"]

    seg_data_array = pose_est_v6._segmentation_dict["seg_data"]

    # test get_segmentation data for each identity
    for i in range(seg_data_array.shape[1]):
        assert np.array_equal(seg_data_array[:, i, ...], pose_est_v6.get_segmentation_data(i))


def test_create_moment(seg_data):
    """Test Moments feature class initialization.

    Verifies that the Moments feature can be properly initialized with
    all expected moment keys.
    """
    pixel_scale = 1.0
    momentsFeature = Moments(seg_data["pose_est_v6"], pixel_scale, seg_data["moment_cache"])

    assert momentsFeature._name == "moments"

    moment_keys = {
        "m00",
        "mu20",
        "mu11",
        "mu02",
        "mu30",
        "mu21",
        "mu12",
        "mu03",
        "nu20",
        "nu11",
        "nu02",
        "nu30",
        "nu21",
        "nu12",
        "nu03",
    }

    assert moment_keys - set(momentsFeature._moments_to_use) == set()


def test_moments_per_frame(seg_data):
    """Test per-frame moment computation.

    Verifies that moments can be computed for each frame and that the
    number of computed moments matches the number of frames in the pose file.
    """
    # initialize moments Feature for first identity
    momentsFeature = seg_data["feature_mods"]["moments"]
    momentValues = momentsFeature.per_frame(1)
    # check that number of moments generated is same as number of frames in pose file
    assert len(momentValues["m00"]) == seg_data["pose_est_v6"].num_frames
