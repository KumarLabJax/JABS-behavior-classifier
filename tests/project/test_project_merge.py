import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import pytest

from jabs.project import Project
from jabs.project.project_merge import MergeStrategy, merge_projects
from jabs.project.video_labels import VideoLabels

# a legacy pose file the converter refuses, so its video is unusable
LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)
UNUSABLE_POSE = LEGACY_CORPUS / "v8-unknown-dataset.h5"
USABLE_POSE = LEGACY_CORPUS / "v8-with-centers.h5"


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
    # source BEHAVIOR inside the destination's NOT_BEHAVIOR block: the only conflict where
    # BEHAVIOR_WINS and DESTINATION_WINS disagree
    src_label.get_track_labels("0", "foo").label_behavior(320, 340)

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
            [(100, 250), (320, 340)],  # expected_behavior
            [(50, 99), (300, 319), (341, 450)],  # expected_not_behavior
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


def test_merge_projects_unique_labeled_video_falls_back_to_direct_pose_path(tmp_path):
    """Unique labeled videos should recover from stale destination inventory via direct pose path."""
    dest = MagicMock()
    src = MagicMock()

    dest.project_paths.project_dir = tmp_path / "dest"
    dest.project_paths.video_dir = dest.project_paths.project_dir / "videos"
    dest.project_paths.pose_dir = dest.project_paths.project_dir / "poses"
    dest.project_paths.cache_dir = dest.project_paths.project_dir / "cache"
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
    dest.video_manager.get_cached_pose_path.side_effect = ValueError("video2.mp4 not in project")

    dest.settings_manager.behavior_names = []
    src.settings_manager.behavior_names = []

    src_labels = VideoLabels("video2.mp4", 100)
    src_labels.get_track_labels("0", "foo").label_behavior(1, 10)
    src.video_manager.load_video_labels.return_value = src_labels

    src_pose = MagicMock()
    src_pose.hash = "hash1"
    src.load_pose_est.return_value = src_pose

    # Destination check_video_name-gated path can fail for unique videos.
    dest.load_pose_est.side_effect = ValueError("video2.mp4 not in project")

    dest_pose = MagicMock()
    dest_pose.hash = "hash1"
    with patch("jabs.project.project_merge.open_pose_file", return_value=dest_pose):
        merge_projects(dest, src, MergeStrategy.DESTINATION_WINS)

    dest.load_pose_est.assert_called_once()
    dest.video_manager.get_cached_pose_path.assert_not_called()
    dest.save_annotations.assert_called_once_with(src_labels, dest_pose)


def _real_project(project_dir: Path, videos: dict[str, Path]) -> "Project":
    """Create a project whose videos are stubs beside copies of the given pose files."""
    project_dir.mkdir(parents=True, exist_ok=True)
    for video, pose in videos.items():
        (project_dir / video).touch()
        shutil.copyfile(pose, project_dir / f"{Path(video).stem}_pose_est_v8.h5")
    return Project(project_dir, enable_session_tracker=False)


def _labels_document(project: "Project", video: str) -> dict:
    """Return the saved annotation document of a video."""
    return json.loads(project.annotation_store.document_path(video).read_text())


def _write_labels(project_dir: Path, video: str, blocks: list[tuple[int, int, bool]]) -> None:
    """Write an annotation document for identity 0 and behavior ``foo``."""
    labels = VideoLabels(video, 120)
    track = labels.get_track_labels("0", "foo")
    for start, end, present in blocks:
        (track.label_behavior if present else track.label_not_behavior)(start, end)
    document = labels.as_dict(DummyPoseEst())
    document["external_identities"] = {"0": "mouse_a"}
    annotations_dir = project_dir / "jabs" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"{Path(video).stem}.json").write_text(json.dumps(document))


def _foo_blocks(document: dict, key: str = "unfragmented_labels") -> list[tuple[int, int, bool]]:
    """Return identity 0's ``foo`` blocks of a document as tuples."""
    return [(b["start"], b["end"], b["present"]) for b in document[key]["0"]["foo"]]


def test_merge_copies_annotations_of_an_unusable_video(tmp_path: Path) -> None:
    """A video whose pose the converter refuses still has its annotations merged."""
    _write_labels(tmp_path / "src", "bad.avi", [(10, 20, True)])
    src = _real_project(tmp_path / "src", {"bad.avi": UNUSABLE_POSE})
    dest = _real_project(tmp_path / "dest", {"ok.avi": USABLE_POSE})
    assert not src.is_usable("bad.avi")

    merge_projects(dest, src, MergeStrategy.BEHAVIOR_WINS)

    merged = Project(tmp_path / "dest", enable_session_tracker=False)
    document = _labels_document(merged, "bad.avi")
    assert _foo_blocks(document) == [(10, 20, True)]
    # no pose to fragment by, so the usable blocks are the labeled blocks
    assert _foo_blocks(document, "labels") == [(10, 20, True)]
    assert document["external_identities"] == {"0": "mouse_a"}


def test_merge_merges_annotations_of_an_unusable_video_in_both_projects(tmp_path: Path) -> None:
    """Both projects label the unusable video: the documents merge by the strategy."""
    _write_labels(tmp_path / "src", "bad.avi", [(10, 20, True)])
    _write_labels(tmp_path / "dest", "bad.avi", [(15, 30, False)])
    src = _real_project(tmp_path / "src", {"bad.avi": UNUSABLE_POSE})
    dest = _real_project(tmp_path / "dest", {"bad.avi": UNUSABLE_POSE})

    merge_projects(dest, src, MergeStrategy.BEHAVIOR_WINS)

    document = _labels_document(dest, "bad.avi")
    assert _foo_blocks(document) == [(10, 20, True), (21, 30, False)]


def test_merge_skips_annotations_of_an_unusable_video_whose_pose_differs(
    tmp_path: Path,
) -> None:
    """The pose-hash check still applies when the pose is unusable."""
    _write_labels(tmp_path / "src", "bad.avi", [(10, 20, True)])
    _write_labels(tmp_path / "dest", "bad.avi", [(15, 30, False)])
    src = _real_project(tmp_path / "src", {"bad.avi": UNUSABLE_POSE})
    dest = _real_project(tmp_path / "dest", {"bad.avi": UNUSABLE_POSE})
    with h5py.File(dest.video_manager.get_cached_pose_path("bad.avi"), "a") as h5:
        h5.attrs["changed"] = 1

    merge_projects(dest, src, MergeStrategy.BEHAVIOR_WINS)

    assert _foo_blocks(_labels_document(dest, "bad.avi")) == [(15, 30, False)]
