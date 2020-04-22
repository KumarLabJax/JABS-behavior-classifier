import unittest

from src.labeler.track_labels import TrackLabels


class TestTrackLabels(unittest.TestCase):
    """ test labeler.track_labels.TrackLabels """

    def test_create(self):
        """
        test initializing new TrackLabels and ensures all frames initialized
        to no label
        """
        labels = TrackLabels(100)
        for i in range(0, 99):
            self.assertEqual(labels.get_frame_label(i), labels.Label.NONE)

    def test_add_behavior_label(self):
        """ test adding a label for a positive observation of the behavior """
        labels = TrackLabels(1000)
        labels.label_behavior(50, 100)

        for i in range(50, 100):
            self.assertEqual(labels.get_frame_label(i), labels.Label.BEHAVIOR)

        # test before and after the block we labeled to make sure it is
        # still unlabeled
        self.assertEqual(labels.get_frame_label(49), labels.Label.NONE)
        self.assertEqual(labels.get_frame_label(101), labels.Label.NONE)

    def test_add_not_behavior_label(self):
        """ test adding a label for a confirmed absence of the behavior """
        labels = TrackLabels(1000)
        labels.label_not_behavior(50, 100)

        for i in range(50, 100):
            self.assertEqual(labels.get_frame_label(i),
                             labels.Label.NOT_BEHAVIOR)

    def test_clear_labels(self):
        """ test clearing labels """
        labels = TrackLabels(100)

        # apply some labels so we can clear them
        labels.label_behavior(15, 30)

        # clear labels we just set
        labels.clear_labels(15, 30)

        # make sure frames no longer have labels
        for i in range(15, 30):
            self.assertEqual(labels.get_frame_label(i),
                             labels.Label.NONE)

    def test_downsample(self):
        """ testing downsampling of label array """
        labels = TrackLabels(75)

        labels.label_behavior(25, 49)
        labels.label_not_behavior(50, 74)

        ds = labels.downsample(3)
        self.assertEqual(ds[0], 0)
        self.assertEqual(ds[1], 1)
        self.assertEqual(ds[2], 2)
        self.assertEqual(len(ds), 3)

    def test_export_behavior_blocks(self):
        """ test exporting to list of label block dicts """
        labels = TrackLabels(1000)
        labels.label_behavior(50, 100)
        labels.label_behavior(195, 205)
        labels.label_not_behavior(215, 250)
        labels.label_behavior(300, 325)

        expected_blocks = [
            {'start': 50, 'end': 100, 'present': True},
            {'start': 195, 'end': 205, 'present': True},
            {'start': 215, 'end': 250, 'present': False},
            {'start': 300, 'end': 325, 'present': True}
        ]

        for exported, expected in zip(labels.get_blocks(), expected_blocks):
            self.assertDictEqual(exported, expected)

    def test_export_behavior_block_slice(self):
        """ test exporting to list of label block dicts """
        labels = TrackLabels(1000)
        labels.label_behavior(0, 100)
        labels.label_behavior(250, 500)

        expected_blocks = [
            {'start': 0, 'end': 25, 'present': True}
        ]

        for exported, expected in zip(labels.get_slice_blocks(0, 25), expected_blocks):
            self.assertDictEqual(exported, expected)

    def test_labeling_single_frame(self):
        """ test labeling a single frame """
        labels = TrackLabels(100)
        labels.label_behavior(25, 25)

        # make sure exactly one frame was labeled
        self.assertEqual(labels.get_frame_label(24), labels.Label.NONE)
        self.assertEqual(labels.get_frame_label(25), labels.Label.BEHAVIOR)
        self.assertEqual(labels.get_frame_label(26), labels.Label.NONE)

        # make sure the block is exported properly
        exported_blocks = labels.get_blocks()
        self.assertEqual(len(exported_blocks), 1)
        self.assertDictEqual({'start': 25, 'end': 25, 'present': True},
                             exported_blocks[0])


if __name__ == '__main__':
    unittest.main()
