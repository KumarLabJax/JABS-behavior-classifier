def test_hu_moment_feature_name(seg_data):
    """Test HuMoment class.

    Verifies that the HuMoments feature has the correct feature names
    and returns the expected number of features per frame.
    """
    # test that data was read and setup correctly
    huMomentsFeature = seg_data["feature_mods"]["hu_moments"]

    assert huMomentsFeature._feature_names[-2] == "hu6"

    i = 0

    huMoments_by_frame = huMomentsFeature.per_frame(i)

    assert len(huMoments_by_frame) == 7
