import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.project import VideoLabels
from jabs.project.video_labels import SERIALIZED_VERSION

mock_pose_est = MagicMock()
mock_pose_est.identity_mask.return_value = np.full(100, 1, dtype=bool)
mock_pose_est.num_frames = 100
mock_pose_est.external_identities = None


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
            "version": SERIALIZED_VERSION,
            "file": "filename.avi",
            "num_frames": 100,
            "labels": {"0": {"behavior name": [{"start": 25, "end": 50, "present": True}]}},
            "unfragmented_labels": {
                "0": {"behavior name": [{"start": 25, "end": 50, "present": True}]}
            },
            "metadata": {
                "project": {},
                "video": {},
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

    def test_iter_identity_behavior_labels(self):
        """Iterating labels should yield each identity/behavior track exactly once."""
        labels = VideoLabels("filename.avi", 100)
        walk_track = labels.get_track_labels("0", "Walk")
        groom_track = labels.get_track_labels("1", "Groom")

        entries = {
            (identity, behavior): track
            for identity, behavior, track in labels.iter_identity_behavior_labels()
        }

        self.assertEqual(set(entries), {("0", "Walk"), ("1", "Groom")})
        self.assertIs(entries[("0", "Walk")], walk_track)
        self.assertIs(entries[("1", "Groom")], groom_track)

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

    def test_unfragmented_labels_not_split_by_identity_gaps(self):
        """test that unfragmented_labels are not fragmented by identity gaps"""
        identity_mask = np.full(1000, True, dtype=bool)
        identity_mask[150:160] = False  # gap from 150 to 159

        mock_pose = MagicMock()
        mock_pose.identity_mask.return_value = identity_mask
        mock_pose.num_frames = 1000

        labels = VideoLabels("test.avi", 1000)
        walking_labels = labels.get_track_labels("0", "Walking")
        walking_labels.label_behavior(100, 200)

        labels_dict = labels.as_dict(mock_pose)
        unfragmented_blocks = labels_dict["unfragmented_labels"]["0"]["Walking"]

        # Should be a single block: 100-200
        expected_unfragmented = [
            {"start": 100, "end": 200, "present": True},
        ]
        self.assertEqual(unfragmented_blocks, expected_unfragmented)

    def test_rename_behavior(self):
        """renaming a behavior should update both labels and unfragmented_labels"""
        labels = VideoLabels("filename.avi", 100)

        # Create an initial behavior with one block
        track = labels.get_track_labels("0", "Walk")
        track.label_behavior(10, 20)

        # Sanity check before rename
        d_before = labels.as_dict(mock_pose_est)
        self.assertIn("Walk", d_before["labels"]["0"])  # old name present
        self.assertIn("Walk", d_before["unfragmented_labels"]["0"])  # old name present

        # Rename the behavior
        labels.rename_behavior("Walk", "Walking")

        # After rename, old key should be gone and new key present in both structures
        d_after = labels.as_dict(mock_pose_est)
        self.assertNotIn("Walk", d_after["labels"]["0"])  # old name removed
        self.assertNotIn("Walk", d_after["unfragmented_labels"]["0"])  # old name removed
        self.assertIn("Walking", d_after["labels"]["0"])  # new name present
        self.assertIn("Walking", d_after["unfragmented_labels"]["0"])  # new name present

        # Ensure the intervals were preserved under the new name
        self.assertEqual(
            d_before["labels"]["0"]["Walk"],
            d_after["labels"]["0"]["Walking"],
        )
        self.assertEqual(
            d_before["unfragmented_labels"]["0"]["Walk"],
            d_after["unfragmented_labels"]["0"]["Walking"],
        )


# ---------------------------------------------------------------------------
# build_multiclass_label_array
# ---------------------------------------------------------------------------

N_FRAMES = 100


@pytest.fixture()
def empty_video_labels() -> VideoLabels:
    """Return a fresh VideoLabels with no annotations."""
    return VideoLabels("test.avi", N_FRAMES)


def test_multiclass_all_unlabeled(empty_video_labels):
    """All frames are 0 when no labels have been applied."""
    result = empty_video_labels.build_multiclass_label_array("0", ["walk", "groom"])
    assert result.shape == (N_FRAMES,)
    assert (result == 0).all()


def test_multiclass_none_label_maps_to_index_1(empty_video_labels):
    """BEHAVIOR on the None track should produce class index 1."""
    track = empty_video_labels.get_track_labels("0", MULTICLASS_NONE_BEHAVIOR)
    track.label_behavior(10, 20)
    result = empty_video_labels.build_multiclass_label_array("0", ["walk", "groom"])
    assert (result[10:21] == 1).all()
    assert (result[:10] == 0).all()
    assert (result[21:] == 0).all()


def test_multiclass_behavior_maps_to_offset_index(empty_video_labels):
    """Behaviors start at index 2 because 0=unlabeled and 1=None are reserved."""
    walk_track = empty_video_labels.get_track_labels("0", "walk")
    walk_track.label_behavior(30, 40)
    groom_track = empty_video_labels.get_track_labels("0", "groom")
    groom_track.label_behavior(60, 70)
    result = empty_video_labels.build_multiclass_label_array("0", ["walk", "groom"])
    assert (result[30:41] == 2).all()
    assert (result[60:71] == 3).all()
    assert (result[:30] == 0).all()


def test_multiclass_only_behavior_frames_labeled(empty_video_labels):
    """NOT_BEHAVIOR frames are treated as unlabeled (class index 0)."""
    track = empty_video_labels.get_track_labels("0", "walk")
    track.label_not_behavior(0, 49)
    track.label_behavior(50, 59)
    result = empty_video_labels.build_multiclass_label_array("0", ["walk"])
    assert (result[:50] == 0).all()
    assert (result[50:60] == 2).all()


def test_multiclass_identity_with_no_labels(empty_video_labels):
    """An identity with no tracks returns an all-zero array."""
    result = empty_video_labels.build_multiclass_label_array("1", ["walk", "groom"])
    assert (result == 0).all()


def test_multiclass_behavior_not_in_list_ignored(empty_video_labels):
    """A labeled behavior absent from behavior_names contributes nothing."""
    track = empty_video_labels.get_track_labels("0", "rear")
    track.label_behavior(5, 15)
    result = empty_video_labels.build_multiclass_label_array("0", ["walk", "groom"])
    assert (result == 0).all()


def test_multiclass_dtype_is_int16(empty_video_labels):
    """Return dtype is int16 regardless of input."""
    result = empty_video_labels.build_multiclass_label_array("0", [])
    assert result.dtype == np.int16


def test_multiclass_reserved_name_raises(empty_video_labels):
    """Passing the reserved MULTICLASS_NONE_BEHAVIOR name raises ValueError."""
    with pytest.raises(ValueError, match="reserved"):
        empty_video_labels.build_multiclass_label_array("0", [MULTICLASS_NONE_BEHAVIOR])


def test_multiclass_duplicate_names_raises(empty_video_labels):
    """Duplicate entries in behavior_names raises ValueError."""
    with pytest.raises(ValueError, match="duplicate"):
        empty_video_labels.build_multiclass_label_array("0", ["walk", "walk"])
