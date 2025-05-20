import unittest

from src.jabs.project import VideoLabels


class TestVideoLabels(unittest.TestCase):
    """test project.video_labels.VideoLabels"""

    def test_create(self):
        """test initializing new VideoLabels"""
        _ = VideoLabels("filename.avi", 100)

    def test_getting_new_track(self):
        """test initializing new VideoLabels"""
        labels = VideoLabels("filename.avi", 100)
        track = labels.get_track_labels("identity name", "behavior name")

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
            "labels": {
                "identity name": {
                    "behavior name": [{"start": 25, "end": 50, "present": True}]
                }
            },
        }

        # create a VideoLabels object from a dict representation
        labels = VideoLabels.load(video_label_dict)

        # check some values of a label track
        track = labels.get_track_labels("identity name", "behavior name")
        self.assertEqual(track.get_frame_label(25), track.Label.BEHAVIOR)
        self.assertEqual(track.get_frame_label(24), track.Label.NONE)
        self.assertEqual(track.get_frame_label(51), track.Label.NONE)

        # make sure exported dict is the same as the one we used to create it
        self.assertDictEqual(labels.as_dict(), video_label_dict)
