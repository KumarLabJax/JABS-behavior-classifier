from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jabs.project.project_merge import MergeStrategy, merge_projects
from jabs.project.video_labels import VideoLabels


@pytest.fixture(autouse=True, scope="session")
def patch_session_tracker():
    """Patch the SessionTracker to avoid side effects during tests."""
    with (
        patch("jabs.project.session_tracker.SessionTracker.__init__", return_value=None),
        patch("jabs.project.session_tracker.SessionTracker.__del__", return_value=None),
    ):
        yield


class DummyPoseEst:
    """Dummy pose estimator for testing purposes."""

    def identity_mask(self, identity):
        """Return a dummy identity mask."""
        return None

    @property
    def external_identities(self):
        """Return a dummy external identities list."""
        return None


@pytest.fixture
def mock_projects_with_labels(tmp_path):
    """Fixture to create two mock projects with video labels for testing merging."""
    dest = MagicMock()
    src = MagicMock()

    dest.project_paths.project_dir = tmp_path / "dest"
    src.project_paths.project_dir = tmp_path / "src"
    dest.project_paths.video_dir = dest.project_paths.project_dir
    dest.project_paths.pose_dir = dest.project_paths.project_dir
    src.project_paths.video_dir = src.project_paths.project_dir
    src.project_paths.pose_dir = src.project_paths.project_dir
    dest.project_paths.project_dir.mkdir()
    src.project_paths.project_dir.mkdir()

    (src.project_paths.project_dir / "video2.mp4").write_bytes(b"dummy video content")
    (src.project_paths.project_dir / "video2_pose_est_v6.h5").write_bytes(b"dummy pose content")

    dest.video_manager.videos = ["video1.mp4"]
    src.video_manager.videos = ["video1.mp4", "video2.mp4"]
    dest.video_manager.video_path.side_effect = lambda v: dest.project_paths.project_dir / v
    src.video_manager.video_path.side_effect = lambda v: src.project_paths.project_dir / v

    dest.settings_manager.behavior_names = ["foo"]
    src.settings_manager.behavior_names = ["foo", "bar"]
    src.settings_manager.get_behavior.side_effect = lambda b: {"name": b}
    dest.settings_manager.save_behavior = MagicMock()

    def pose_obj(hash_val):
        obj = MagicMock()
        obj.hash = hash_val
        return obj

    dest.load_pose_est.side_effect = lambda p: pose_obj("hash1")
    src.load_pose_est.side_effect = lambda p: pose_obj("hash1")
    src.video_manager.get_cached_pose_path.side_effect = (
        lambda v: src.project_paths.pose_dir / f"{Path(v).stem}_pose_est_v6.h5"
    )

    dest_label = VideoLabels("video1.mp4", 1000)
    src_label = VideoLabels("video1.mp4", 1000)

    dest_label.get_track_labels("0", "foo").label_behavior(100, 200)
    dest_label.get_track_labels("0", "foo").label_not_behavior(300, 400)
    src_label.get_track_labels("0", "foo").label_behavior(150, 250)
    src_label.get_track_labels("0", "foo").label_not_behavior(50, 150)
    src_label.get_track_labels("0", "foo").label_not_behavior(350, 450)

    dest.video_manager.load_video_labels.side_effect = (
        lambda v: dest_label if v == "video1.mp4" else None
    )
    src.video_manager.load_video_labels.side_effect = (
        lambda v: src_label if v == "video1.mp4" else MagicMock()
    )
    dest.save_annotations = MagicMock()

    yield dest, src, dest_label, src_label


@pytest.mark.parametrize(
    "strategy,expected_behavior,expected_not_behavior",
    [
        (
            MergeStrategy.BEHAVIOR_WINS,
            [(100, 250)],  # expected_behavior
            [(50, 99), (300, 450)],  # expected_not_behavior
        ),
        (
            MergeStrategy.NOT_BEHAVIOR_WINS,
            [(151, 250)],
            [(50, 150), (300, 450)],
        ),
        (
            MergeStrategy.DESTINATION_WINS,
            [(100, 250)],
            [(50, 99), (300, 450)],
        ),
    ],
)
def test_merge_projects_label_merging(
    mock_projects_with_labels, strategy, expected_behavior, expected_not_behavior
):
    """Test merging two projects with labels using different strategies."""
    dest, src, dest_label, src_label = mock_projects_with_labels

    merge_projects(dest, src, strategy)

    saved_labels = [
        call_args[0][0]
        for call_args in dest.save_annotations.call_args_list
        if isinstance(call_args[0][0], VideoLabels) and call_args[0][0].filename == "video1.mp4"
    ]
    assert saved_labels, "No merged labels saved for video1"
    merged_label = saved_labels[0]

    pose_est = DummyPoseEst()
    merged_dict = merged_label.as_dict(pose_est)

    expected = VideoLabels("video1.mp4", 1000)
    track = expected.get_track_labels("0", "foo")
    for start, end in expected_behavior:
        track.label_behavior(start, end)
    for start, end in expected_not_behavior:
        track.label_not_behavior(start, end)
    expected_dict = expected.as_dict(pose_est)

    def sort_blocks(d):
        # Recursively sort lists of dicts by 'start' and 'end'
        if isinstance(d, dict):
            return {k: sort_blocks(v) for k, v in d.items()}
        if isinstance(d, list) and d and isinstance(d[0], dict):
            return sorted([sort_blocks(x) for x in d], key=lambda x: (x["start"], x["end"]))
        return d

    assert sort_blocks(merged_dict) == sort_blocks(expected_dict)


def test_merge_projects_copies_unique_video_and_pose_to_split_dirs(tmp_path):
    """Unique source videos should be copied into the destination's configured video and pose dirs."""
    dest = MagicMock()
    src = MagicMock()

    dest.project_paths.project_dir = tmp_path / "dest"
    dest.project_paths.video_dir = dest.project_paths.project_dir / "videos"
    dest.project_paths.pose_dir = dest.project_paths.project_dir / "poses"
    src.project_paths.project_dir = tmp_path / "src"
    src.project_paths.video_dir = src.project_paths.project_dir / "videos"
    src.project_paths.pose_dir = src.project_paths.project_dir / "poses"

    for path in (
        dest.project_paths.video_dir,
        dest.project_paths.pose_dir,
        src.project_paths.video_dir,
        src.project_paths.pose_dir,
    ):
        path.mkdir(parents=True)

    (src.project_paths.video_dir / "video2.mp4").write_bytes(b"dummy video content")
    (src.project_paths.pose_dir / "video2_pose_est_v6.h5").write_bytes(b"dummy pose content")

    dest.video_manager.videos = []
    src.video_manager.videos = ["video2.mp4"]
    src.video_manager.video_path.side_effect = lambda v: src.project_paths.video_dir / v
    src.video_manager.get_cached_pose_path.side_effect = (
        lambda v: src.project_paths.pose_dir / f"{Path(v).stem}_pose_est_v6.h5"
    )
    src.video_manager.load_video_labels.return_value = None
    dest.settings_manager.behavior_names = []
    src.settings_manager.behavior_names = []

    merge_projects(dest, src, MergeStrategy.DESTINATION_WINS)

    assert (dest.project_paths.video_dir / "video2.mp4").read_bytes() == b"dummy video content"
    assert (dest.project_paths.pose_dir / "video2_pose_est_v6.h5").read_bytes() == (
        b"dummy pose content"
    )
