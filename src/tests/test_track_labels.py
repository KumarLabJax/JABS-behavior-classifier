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
            self.assertEqual(labels.get_frame_label(50), labels.Label.BEHAVIOR)

        # test before and after the block we labeled to make sure it is
        # still unlabeled
        self.assertEqual(labels.get_frame_label(49), labels.Label.NONE)
        self.assertEqual(labels.get_frame_label(101), labels.Label.NONE)

    def test_add_not_behavior_label(self):
        """ test adding a label for a confirmed absence of the behavior """
        labels = TrackLabels(1000)
        labels.label_not_behavior(50, 100)

        for i in range(50, 100):
            self.assertEqual(labels.get_frame_label(50),
                             labels.Label.NOT_BEHAVIOR)

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

        for exported, expected in zip(labels.export(), expected_blocks):
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
        exported_blocks = labels.export()
        self.assertEqual(len(exported_blocks), 1)
        self.assertDictEqual({'start': 25, 'end': 25, 'present': True},
                             exported_blocks[0])


if __name__ == '__main__':
    unittest.main()
