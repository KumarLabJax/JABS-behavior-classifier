"""Annotation operations on a project with an unusable, annotated video.

Archive, rename and the overlap query work on annotations, which are JSON and
do not need the pose file, so an unusable video must neither block them nor
cost labels.
"""

import gzip
import json
import logging
import shutil
from pathlib import Path

import pytest

from jabs.project import Project
from jabs.project.video_labels import VideoLabels

# a legacy pose file the converter refuses, so its video is unusable
LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)
UNUSABLE_POSE = LEGACY_CORPUS / "v8-unknown-dataset.h5"
USABLE_POSE = LEGACY_CORPUS / "v8-with-centers.h5"


class _NoMaskPose:
    """Stand-in pose for writing a document: no identity mask, no external ids."""

    external_identities = None

    def identity_mask(self, identity: int) -> None:
        """Return no mask, so the stored blocks are the labeled blocks."""
        return None


def _write_labels(project_dir: Path, video: str, behaviors: dict[str, tuple[int, int]]) -> None:
    """Write an annotation document labeling identity 0 for each behavior's frame range."""
    labels = VideoLabels(video, 120)
    for behavior, (start, end) in behaviors.items():
        labels.get_track_labels("0", behavior).label_behavior(start, end)
    document = labels.as_dict(_NoMaskPose())
    document["external_identities"] = {"0": "mouse_a"}
    annotations_dir = project_dir / "jabs" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"{Path(video).stem}.json").write_text(json.dumps(document))


def _document(project: Project, video: str) -> dict:
    """Return the saved annotation document of a video."""
    return json.loads(project.annotation_store.document_path(video).read_text())


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """A project whose usable ``a.avi`` comes before its unusable ``b.avi``; both labeled."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    for video, pose in (("a.avi", USABLE_POSE), ("b.avi", UNUSABLE_POSE)):
        (project_dir / video).touch()
        shutil.copyfile(pose, project_dir / f"{Path(video).stem}_pose_est_v8.h5")
        _write_labels(project_dir, video, {"foo": (10, 20), "bar": (30, 40)})
    result = Project(project_dir, enable_session_tracker=False)
    for behavior in ("foo", "bar"):
        result.settings_manager.save_behavior(behavior, {})
    assert result.is_usable("a.avi")
    assert not result.is_usable("b.avi")
    return result


def test_archive_behavior_keeps_labels_of_an_unusable_video(project: Project) -> None:
    """Archiving moves both videos' labels into the archive; nothing is lost."""
    project.archive_behavior("foo")

    archives = list(project.project_paths.archive_dir.glob("foo_*.json.gz"))
    assert len(archives) == 1
    with gzip.open(archives[0], "rt") as f:
        archived = json.load(f)
    for video in ("a.avi", "b.avi"):
        assert archived[video]["unfragmented_labels"]["foo"]["0"] == [
            {"start": 10, "end": 20, "present": True}
        ]
        document = _document(project, video)
        assert "foo" not in document["unfragmented_labels"]["0"]
        assert "foo" not in document["labels"]["0"]
        assert "bar" in document["unfragmented_labels"]["0"]
    # the unusable video's document is edited in place, so it keeps what only
    # the pose could otherwise supply
    assert _document(project, "b.avi")["external_identities"] == {"0": "mouse_a"}
    assert "foo" not in project.settings_manager.behavior_names


def test_rename_behavior_renames_labels_of_an_unusable_video(project: Project) -> None:
    """Renaming reaches the unusable video's labels and the project settings."""
    project.rename_behavior("foo", "baz")

    for video in ("a.avi", "b.avi"):
        document = _document(project, video)
        for section in ("labels", "unfragmented_labels"):
            assert "foo" not in document[section]["0"]
            assert document[section]["0"]["baz"] == [{"start": 10, "end": 20, "present": True}]
    assert _document(project, "b.avi")["external_identities"] == {"0": "mouse_a"}
    assert "baz" in project.settings_manager.behavior_names
    assert "foo" not in project.settings_manager.behavior_names


def test_rename_behavior_conflict_changes_nothing(project: Project) -> None:
    """A rename refused for one video's labels leaves every video and the settings as they were."""
    document = _document(project, "b.avi")
    document["unfragmented_labels"]["0"]["baz"] = [{"start": 50, "end": 60, "present": True}]
    project.annotation_store.document_path("b.avi").write_text(json.dumps(document))
    before = {video: _document(project, video) for video in ("a.avi", "b.avi")}

    with pytest.raises(KeyError):
        project.rename_behavior("foo", "baz")

    assert {video: _document(project, video) for video in ("a.avi", "b.avi")} == before
    assert "foo" in project.settings_manager.behavior_names


def test_overlap_query_reads_labels_of_an_unusable_video(project: Project) -> None:
    """The overlap query finds a conflict in an unusable video's labels."""
    document = _document(project, "b.avi")
    document["unfragmented_labels"]["0"]["bar"] = [{"start": 15, "end": 25, "present": True}]
    project.annotation_store.document_path("b.avi").write_text(json.dumps(document))

    assert project.get_overlapping_behavior_label_videos() == ["b.avi"]


def test_unusable_video_is_not_trainable(project: Project) -> None:
    """Training and its readiness counts leave out an unusable video."""
    assert project.is_trainable("a.avi")
    assert not project.is_trainable("b.avi")
    project.settings_manager.set_video_excluded("a.avi", True)
    assert not project.is_trainable("a.avi")


def test_project_unusable_videos_notice_flags_labeled_videos(project: Project) -> None:
    """The project summary lists the unusable video as labeled."""
    notice = project.unusable_videos_notice()

    assert notice is not None
    assert notice.details == "b.avi (labeled): poseest/mystery is not a legacy pose dataset"


def test_project_skipped_training_notice_only_for_labeled_behaviors(project: Project) -> None:
    """A training run is told about an unusable video only when it has labels for it."""
    notice = project.skipped_training_notice(["foo"])
    assert notice is not None
    assert notice.details.startswith("b.avi:")
    assert project.skipped_training_notice(["baz"]) is None


def test_warn_unusable_videos_logs_one_summary(
    project: Project, caplog: pytest.LogCaptureFixture
) -> None:
    """Command-line tools get one WARNING naming the unusable video and its reason."""
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        project.warn_unusable_videos()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "b.avi (labeled): poseest/mystery" in warnings[0].getMessage()
