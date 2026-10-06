import unittest

import numpy as np

from jabs.project.track_labels import TrackLabels


class TestTrackLabels(unittest.TestCase):
    """test project.track_labels.TrackLabels"""

    def test_create(self):
        """test initializing new TrackLabels

        ensures all frames initialized to no label
        """
        labels = TrackLabels(100)
        self.assertEqual(len(labels.get_labels()), 100)
        for i in range(100):
            self.assertEqual(labels.get_frame_label(i), labels.Label.NONE)

    def test_add_behavior_label(self):
        """test adding a label for a positive observation of the behavior"""
        labels = TrackLabels(1000)
        labels.label_behavior(50, 100)

        # the end frame is inclusive
        for i in range(50, 101):
            self.assertEqual(labels.get_frame_label(i), labels.Label.BEHAVIOR)

        # test before and after the block we labeled to make sure it is
        # still unlabeled
        self.assertEqual(labels.get_frame_label(49), labels.Label.NONE)
        self.assertEqual(labels.get_frame_label(101), labels.Label.NONE)

    def test_add_not_behavior_label(self):
        """test adding a label for a confirmed absence of the behavior"""
        labels = TrackLabels(1000)
        labels.label_not_behavior(50, 100)

        # the end frame is inclusive
        for i in range(50, 101):
            self.assertEqual(labels.get_frame_label(i), labels.Label.NOT_BEHAVIOR)

        # frames before and after the block we labeled are still unlabeled
        self.assertEqual(labels.get_frame_label(49), labels.Label.NONE)
        self.assertEqual(labels.get_frame_label(101), labels.Label.NONE)

    def test_clear_labels(self):
        """test clearing labels"""
        labels = TrackLabels(100)

        # apply some labels so we can clear them
        labels.label_behavior(10, 40)

        # clear part of the labels we just set, [15, 30] inclusive
        labels.clear_labels(15, 30)

        # frames in the cleared range, including its end frame, no longer have labels
        for i in range(15, 31):
            self.assertEqual(labels.get_frame_label(i), labels.Label.NONE)

        # frames on either side of the cleared range keep their labels
        for i in [*range(10, 15), *range(31, 41)]:
            self.assertEqual(labels.get_frame_label(i), labels.Label.BEHAVIOR)

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

        self.assertListEqual(labels.get_blocks(), expected_blocks)

    def test_export_behavior_block_slice(self):
        """test exporting to list of label block dicts"""
        labels = TrackLabels(1000)
        labels.label_behavior(0, 100)
        labels.label_behavior(250, 500)

        expected_blocks = [{"start": 0, "end": 25, "present": True}]
        self.assertListEqual(labels.get_slice_blocks(0, 25), expected_blocks)

        # block frame numbers are relative to the slice start, and the slice end is inclusive
        expected_blocks = [
            {"start": 0, "end": 20, "present": True},
            {"start": 170, "end": 190, "present": True},
        ]
        self.assertListEqual(labels.get_slice_blocks(80, 270), expected_blocks)

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


def _labelled_track(num_frames: int = 6) -> TrackLabels:
    """Return a track with the first half behavior and the second half not-behavior."""
    track = TrackLabels(num_frames)
    track.label_behavior(0, num_frames // 2 - 1)
    track.label_not_behavior(num_frames // 2, num_frames - 1)
    return track


def test_labels_masked_to_identity_clears_absent_frames() -> None:
    """Frames the identity is missing from cannot carry a label."""
    track = _labelled_track(6)
    identity_mask = np.array([1, 1, 0, 0, 1, 1])

    masked = track.labels_masked_to_identity(identity_mask)

    assert masked.tolist() == [
        TrackLabels.Label.BEHAVIOR,
        TrackLabels.Label.BEHAVIOR,
        TrackLabels.Label.NONE,
        TrackLabels.Label.NONE,
        TrackLabels.Label.NOT_BEHAVIOR,
        TrackLabels.Label.NOT_BEHAVIOR,
    ]


def test_labels_masked_to_identity_does_not_mutate_the_stored_labels() -> None:
    """The result is a copy, so callers cannot clear the stored labels by accident.

    ``get_labels()`` returns the underlying array by reference, so masking in
    place through it used to rewrite the annotations the caller was reading.
    """
    track = _labelled_track(6)
    before = track.get_labels().copy()

    masked = track.labels_masked_to_identity(np.zeros(6, dtype=bool))

    assert masked.tolist() == [TrackLabels.Label.NONE] * 6
    assert track.get_labels().tolist() == before.tolist()
    assert masked is not track.get_labels()


def test_labels_masked_to_identity_accepts_an_integer_mask() -> None:
    """``PoseEstimation.identity_mask`` yields integers, not booleans."""
    track = _labelled_track(4)
    as_int = track.labels_masked_to_identity(np.array([1, 0, 1, 0], dtype=np.int64))
    as_bool = track.labels_masked_to_identity(np.array([True, False, True, False]))

    assert as_int.tolist() == as_bool.tolist()


def test_labels_masked_to_identity_leaves_a_fully_present_identity_alone() -> None:
    """An identity present in every frame keeps all of its labels."""
    track = _labelled_track(6)

    masked = track.labels_masked_to_identity(np.ones(6, dtype=bool))

    assert masked.tolist() == track.get_labels().tolist()


if __name__ == "__main__":
    unittest.main()
