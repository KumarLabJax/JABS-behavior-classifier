import unittest

# project imports
from .seg_test_utils import SegDataBaseClass as SBC
from jabs.feature_extraction.segmentation_features import HuMoments


class Test(SBC, unittest.TestCase):
    """This test will provide testing coverage for the HuMoments Feature class."""

    def testHuMomentFeatureName(self) -> None:
        """Test HuMoment class."""

        # test that data was read and setup correctly
        huMomentsFeature = self.feature_mods["hu_moments"]

        assert huMomentsFeature._feature_names[-2] == "hu6"

        i = 0

        huMoments_by_frame = huMomentsFeature.per_frame(i)

        assert len(huMoments_by_frame) == 7
