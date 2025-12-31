import numpy as np

from jabs.feature_extraction.segmentation_features import Moments


def test_data(seg_data):
    """Test segmentation data integrity.

    Verifies that segmentation data has proper shape and is non-empty.
    """
    seg_data_array = seg_data["pose_est_v6"]._segmentation_dict["seg_data"]

    assert len(seg_data_array.shape) == 5
    assert sum(seg_data_array.shape) > 0


def test_posev6_instantiation(seg_data):
    """Test PoseEstimationV6 class instantiation with segmentation data.

    Verifies that pose estimation v6 object is properly created with
    segmentation dictionary and non-trivial segmentation data.
    """
    pose_est_v6 = seg_data["pose_est_v6"]

    # test that pose estimation object was created, and that a segmentation_dict attribute is present.
    assert hasattr(pose_est_v6, "_segmentation_dict")

    seg_data_array = pose_est_v6._segmentation_dict["seg_data"]

    # non-empty segmentation data
    assert sum(seg_data_array.shape) > 0

    # non-trivial data
    assert len(np.unique(seg_data_array)) > 1

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
