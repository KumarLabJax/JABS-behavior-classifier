import unittest
from unittest.mock import MagicMock

import numpy as np

from src.jabs.project import VideoLabels

mock_pose_est = MagicMock()
mock_pose_est.identity_mask.return_value = np.full(100, 1, dtype=bool)
mock_pose_est.num_frames = 100


class TestVideoLabels(unittest.TestCase):
    """test project.video_labels.VideoLabels"""

    def test_create(self):
        """test initializing new VideoLabels"""
        _ = VideoLabels("filename.avi", mock_pose_est)

    def test_getting_new_track(self):
        """test initializing new VideoLabels"""
        labels = VideoLabels("filename.avi", 100)
        track = labels.get_track_labels("0", "behavior name")

        for i in range(0, 99):
            self.assertEqual(track.get_frame_label(i), track.Label.NONE)

    def test_identity_must_be_string(self):
        """Test that identity must be a string."""
        labels = VideoLabels("filename.avi", 100)

        with self.assertRaises(ValueError):
            _ = labels.get_track_labels(1, "behavior name")

    def test_load_from_dict(self):
        """Test creating new VideoLabels object from dict representation."""
        video_label_dict = {
            "file": "filename.avi",
            "num_frames": 100,
            "labels": {"0": {"behavior name": [{"start": 25, "end": 50, "present": True}]}},
            "unfragmented_labels": {
                "0": {"behavior name": [{"start": 25, "end": 50, "present": True}]}
            },
        }

        # create a VideoLabels object from a dict representation
        labels = VideoLabels.load(video_label_dict)

        # check some values of a label track
        track = labels.get_track_labels("0", "behavior name")
        self.assertEqual(track.get_frame_label(25), track.Label.BEHAVIOR)
        self.assertEqual(track.get_frame_label(24), track.Label.NONE)
        self.assertEqual(track.get_frame_label(51), track.Label.NONE)

        # make sure exported dict is the same as the one we used to create it
        self.assertDictEqual(labels.as_dict(mock_pose_est), video_label_dict)

    def test_label_fragmentation_with_identity_gaps(self):
        """test that label blocks are fragmented when there are identity gaps"""
        # Create a mask with a gap in the interval 100-200
        identity_mask = np.full(1000, True, dtype=bool)
        identity_mask[150:160] = False  # gap from 150 to 159

        mock_pose = MagicMock()
        mock_pose.identity_mask.return_value = identity_mask
        mock_pose.num_frames = 1000

        labels = VideoLabels("test.avi", 1000)
        walking_labels = labels.get_track_labels("0", "Walking")
        walking_labels.label_behavior(100, 200)

        labels_dict = labels.as_dict(mock_pose)
        label_blocks = labels_dict["labels"]["0"]

        # Should be two blocks: 100-149 and 160-200
        expected_blocks = [
            {"start": 100, "end": 149, "present": True},
            {"start": 160, "end": 200, "present": True},
        ]
        self.assertEqual(label_blocks["Walking"], expected_blocks)
