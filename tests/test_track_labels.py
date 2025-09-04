import unittest

import numpy as np

from jabs.project.track_labels import TrackLabels


class TestTrackLabels(unittest.TestCase):
    """test project.track_labels.TrackLabels"""

    def test_create(self):
        """
        test initializing new TrackLabels and ensures all frames initialized
        to no label
        """
        labels = TrackLabels(100)
        for i in range(0, 99):
            self.assertEqual(labels.get_frame_label(i), labels.Label.NONE)

    def test_add_behavior_label(self):
        """test adding a label for a positive observation of the behavior"""
        labels = TrackLabels(1000)
        labels.label_behavior(50, 100)

        for i in range(50, 100):
            self.assertEqual(labels.get_frame_label(i), labels.Label.BEHAVIOR)

        # test before and after the block we labeled to make sure it is
        # still unlabeled
        self.assertEqual(labels.get_frame_label(49), labels.Label.NONE)
        self.assertEqual(labels.get_frame_label(101), labels.Label.NONE)

    def test_add_not_behavior_label(self):
        """test adding a label for a confirmed absence of the behavior"""
        labels = TrackLabels(1000)
        labels.label_not_behavior(50, 100)

        for i in range(50, 100):
            self.assertEqual(labels.get_frame_label(i), labels.Label.NOT_BEHAVIOR)

    def test_clear_labels(self):
        """test clearing labels"""
        labels = TrackLabels(100)

        # apply some labels so we can clear them
        labels.label_behavior(15, 30)

        # clear labels we just set
        labels.clear_labels(15, 30)

        # make sure frames no longer have labels
        for i in range(15, 30):
            self.assertEqual(labels.get_frame_label(i), labels.Label.NONE)

    def test_downsample_basic(self):
        """testing downsampling of label array"""
        # downsample into an array of length 3
        # will result in three bins of uniform value
        # 0-24 will be unlabeled
        # 25-49 labeled with behavior
        # 50-74 labeled not behavior
        labels = TrackLabels(75)
        labels.label_behavior(25, 49)
        labels.label_not_behavior(50, 74)

        # downsample into an array of length 3
        ds = TrackLabels.downsample(labels.get_labels(), 3)

        # confirm length
        self.assertEqual(len(ds), 3)

        # we should end up with tree bins, each with a different value
        self.assertEqual(ds[0], TrackLabels.Label.NONE)
        self.assertEqual(ds[1], TrackLabels.Label.BEHAVIOR)
        self.assertEqual(ds[2], TrackLabels.Label.NOT_BEHAVIOR)

    def test_downsample_mixed_1(self):
        """
        test that a bin containing mix of Label.NONE and Label.BEHAVIOR results
        in a value of Label.BEHAVIOR in downsampled array
        """
        labels = TrackLabels(10)

        # label position 0, 1, 2
        labels.label_behavior(0, 2)

        # downsample into an array of length two
        # ds[0] will be computed from [1, 1, 1, 0, 0]
        # ds[1] will be computed from [0, 0, 0, 0, 0]
        ds = TrackLabels.downsample(labels.get_labels(), 2)

        self.assertEqual(ds[0], TrackLabels.Label.BEHAVIOR)
        self.assertEqual(ds[1], TrackLabels.Label.NONE)

    def test_downsample_mixed_2(self):
        """
        test that a bin containing mix of Label.BEHAVIOR and Label.NOT_BEHAVIOR
        results in a special value (Label.MIX)
        """
        labels = TrackLabels(10)

        # label position 0, 1,
        labels.label_behavior(0, 1)
        labels.label_not_behavior(2, 3)

        # downsample into an array of length two
        # ds[0] will be computed from [1, 1, 2, 2, 0]
        # ds[1] will be computed from [0, 0, 0, 0, 0]
        ds = TrackLabels.downsample(labels.get_labels(), 2)

        self.assertEqual(ds[0], TrackLabels.Label.MIX)
        self.assertEqual(ds[1], TrackLabels.Label.NONE)

    def test_downsample_non_divisible(self):
        """
        test that we can downsample to a size that doesn't evenly divide
        the label array
        """
        labels = TrackLabels(100)
        ds = TrackLabels.downsample(labels.get_labels(), 33)

        self.assertEqual(len(ds), 33)

    def test_export_behavior_blocks(self):
        """test exporting to list of label block dicts"""
        labels = TrackLabels(1000)
        labels.label_behavior(50, 100)
        labels.label_behavior(195, 205)
        labels.label_not_behavior(215, 250)
        labels.label_behavior(300, 325)

        expected_blocks = [
            {"start": 50, "end": 100, "present": True},
            {"start": 195, "end": 205, "present": True},
            {"start": 215, "end": 250, "present": False},
            {"start": 300, "end": 325, "present": True},
        ]

        for exported, expected in zip(labels.get_blocks(), expected_blocks, strict=False):
            self.assertDictEqual(exported, expected)

    def test_export_behavior_block_slice(self):
        """test exporting to list of label block dicts"""
        labels = TrackLabels(1000)
        labels.label_behavior(0, 100)
        labels.label_behavior(250, 500)

        expected_blocks = [{"start": 0, "end": 25, "present": True}]

        for exported, expected in zip(
            labels.get_slice_blocks(0, 25), expected_blocks, strict=False
        ):
            self.assertDictEqual(exported, expected)

    def test_labeling_single_frame(self):
        """test labeling a single frame"""
        labels = TrackLabels(100)
        labels.label_behavior(25, 25)

        # make sure exactly one frame was labeled
        self.assertEqual(labels.get_frame_label(24), labels.Label.NONE)
        self.assertEqual(labels.get_frame_label(25), labels.Label.BEHAVIOR)
        self.assertEqual(labels.get_frame_label(26), labels.Label.NONE)

        # make sure the block is exported properly
        exported_blocks = labels.get_blocks()
        self.assertEqual(len(exported_blocks), 1)
        self.assertDictEqual({"start": 25, "end": 25, "present": True}, exported_blocks[0])

    def test_label_with_mask(self):
        """test labeling with a mask"""
        labels = TrackLabels(10)
        # labels should only get applied where mask value is 1
        mask = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        labels.label_behavior(0, 9, mask=mask)

        # make sure locations with mask 0 were not labeled
        expected_val = np.full(10, labels.Label.NONE.value, dtype="int")
        expected_val[5:10] = labels.Label.BEHAVIOR
        self.assertListEqual(list(expected_val), list(labels.get_labels()))


if __name__ == "__main__":
    unittest.main()
