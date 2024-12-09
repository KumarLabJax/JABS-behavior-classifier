import unittest

# project imports
from .seg_test_utils import SegDataBaseClass as SBC
from src.jabs.feature_extraction.segmentation_features import HuMoments


class Test(SBC, unittest.TestCase):
    """ This test will provide testing coverage for the HuMoments Feature class. """

    pixel_scale = 1.0

    def testHuMomentFeatureName(self) -> None:
        """ Test HuMoment class. """

        # test that data was read and setup correctly
        seg_data = self._pose_est_v6._segmentation_dict["seg_data"]
        assert sum(self.seg_data.shape) == sum(seg_data.shape)

        huMomentsFeature = HuMoments(self._pose_est_v6, self.pixel_scale)
        
        assert huMomentsFeature._feature_names[-2] == "hu6"

        i = 0

        huMoments_by_frame = huMomentsFeature.per_frame(i)

        assert huMoments_by_frame.shape[1] == 7

        

