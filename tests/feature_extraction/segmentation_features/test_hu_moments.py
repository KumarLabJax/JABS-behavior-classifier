import numpy as np


def test_hu_moment_feature_name(seg_data):
    """Test HuMoment class.

    Verifies that the HuMoments feature has the correct feature names, returns the
    expected number of features per frame, and fills them in for frames with a shape.
    """
    # test that data was read and setup correctly
    huMomentsFeature = seg_data["feature_mods"]["hu_moments"]

    assert huMomentsFeature._feature_names[-2] == "hu6"

    # per_frame ignores its identity argument: the fixture bound these feature modules to
    # identity 1's moment cache, so identity 1 is also the one to read m00 for below.
    identity = 1

    huMoments_by_frame = huMomentsFeature.per_frame(identity)

    assert len(huMoments_by_frame) == 7

    # hu1 is nu20 + nu02, which is positive for any shape with area, so a feature left at
    # its zero fill or filled with NaN fails here.
    m00 = seg_data["feature_mods"]["moments"].per_frame(identity)["m00"]
    has_shape = m00 > 0
    assert has_shape.any()
    hu1 = huMoments_by_frame["hu1"][has_shape]
    assert np.isfinite(hu1).all()
    assert (hu1 != 0).all()
