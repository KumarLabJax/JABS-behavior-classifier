"""Tests for selecting unlabeled videos to prune from a project."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode
from jabs.project.project_pruning import VideoPaths, get_videos_to_prune

EMPTY_COUNTS: dict = {}
LABELED_COUNTS = {
    0: {
        "fragmented_frame_counts": (10, 0),
        "fragmented_bout_counts": (1, 0),
        "unfragmented_frame_counts": (10, 0),
        "unfragmented_bout_counts": (1, 0),
    }
}
ZERO_COUNTS = {
    0: {
        "fragmented_frame_counts": (0, 0),
        "fragmented_bout_counts": (0, 0),
        "unfragmented_frame_counts": (0, 0),
        "unfragmented_bout_counts": (0, 0),
    }
}


def _make_project(
    mode: ClassifierMode,
    behavior_names: list[str],
    counts: dict[tuple[str, str], dict],
    videos: list[str],
) -> MagicMock:
    """Build a Project-like mock whose label counts come from ``counts``.

    Args:
        mode: classifier mode reported by the settings manager.
        behavior_names: behaviors defined in the project (excludes the None track).
        counts: mapping of ``(video, behavior)`` to the dict ``load_counts`` returns.
        videos: video names in the project.

    Returns:
        A mock with the attributes ``get_videos_to_prune`` uses.
    """
    project = MagicMock()
    project.settings_manager = SimpleNamespace(
        classifier_mode=mode, behavior_names=list(behavior_names)
    )
    project.video_manager.videos = videos
    project.video_manager.video_path.side_effect = lambda v: Path(v)
    project.video_manager.get_cached_pose_path.side_effect = lambda v: Path(f"{v}_pose.h5")
    project.annotation_store.document_path.side_effect = lambda v: Path(f"{v}.json")
    project.load_counts.side_effect = lambda video, behavior: counts.get(
        (video, behavior), EMPTY_COUNTS
    )
    return project


def _pruned_names(result: list[VideoPaths]) -> list[str]:
    return [r.video_path.name for r in result]


def test_binary_prunes_unlabeled_video() -> None:
    """A binary project prunes videos with no labels for any behavior."""
    project = _make_project(
        ClassifierMode.BINARY,
        ["Walk"],
        {("a.avi", "Walk"): LABELED_COUNTS},
        ["a.avi", "b.avi"],
    )

    assert _pruned_names(get_videos_to_prune(project)) == ["b.avi"]


def test_zero_counts_are_unlabeled() -> None:
    """Counts that are all zero do not count as labels."""
    project = _make_project(
        ClassifierMode.BINARY, ["Walk"], {("a.avi", "Walk"): ZERO_COUNTS}, ["a.avi"]
    )

    assert _pruned_names(get_videos_to_prune(project)) == ["a.avi"]


def test_binary_mode_ignores_none_track() -> None:
    """The None track only exists in multi-class mode, so binary projects don't consult it."""
    project = _make_project(
        ClassifierMode.BINARY,
        ["Walk"],
        {("a.avi", MULTICLASS_NONE_BEHAVIOR): LABELED_COUNTS},
        ["a.avi"],
    )

    assert _pruned_names(get_videos_to_prune(project)) == ["a.avi"]


def test_multiclass_keeps_video_with_only_none_labels() -> None:
    """Regression test for KLAUS-726: None-only videos are training data, not unlabeled."""
    project = _make_project(
        ClassifierMode.MULTICLASS,
        ["Walk", "Run"],
        {("none_only.avi", MULTICLASS_NONE_BEHAVIOR): LABELED_COUNTS},
        ["none_only.avi", "empty.avi"],
    )

    assert _pruned_names(get_videos_to_prune(project)) == ["empty.avi"]


def test_multiclass_keeps_video_with_only_behavior_labels() -> None:
    """Multi-class videos labeled only on a behavior track are kept."""
    project = _make_project(
        ClassifierMode.MULTICLASS,
        ["Walk", "Run"],
        {("run.avi", "Run"): LABELED_COUNTS},
        ["run.avi", "empty.avi"],
    )

    assert _pruned_names(get_videos_to_prune(project)) == ["empty.avi"]


def test_multiclass_prunes_video_with_no_labels_on_any_track() -> None:
    """Multi-class videos with no labels on any track are pruned."""
    project = _make_project(ClassifierMode.MULTICLASS, ["Walk"], {}, ["a.avi"])

    assert _pruned_names(get_videos_to_prune(project)) == ["a.avi"]


def test_multiclass_without_behaviors_still_checks_none_track() -> None:
    """The None track is checked even when the project has no behaviors."""
    project = _make_project(
        ClassifierMode.MULTICLASS,
        [],
        {("a.avi", MULTICLASS_NONE_BEHAVIOR): LABELED_COUNTS},
        ["a.avi"],
    )

    assert get_videos_to_prune(project) == []


def test_specific_behavior_only_checks_that_behavior() -> None:
    """An explicit behavior argument is honored as-is, even in multi-class mode."""
    project = _make_project(
        ClassifierMode.MULTICLASS,
        ["Walk", "Run"],
        {
            ("walk.avi", "Walk"): LABELED_COUNTS,
            ("none_only.avi", MULTICLASS_NONE_BEHAVIOR): LABELED_COUNTS,
        },
        ["walk.avi", "none_only.avi"],
    )

    assert _pruned_names(get_videos_to_prune(project, "Walk")) == ["none_only.avi"]


@pytest.mark.parametrize("mode", [ClassifierMode.BINARY, ClassifierMode.MULTICLASS])
def test_video_paths_are_populated(mode: ClassifierMode) -> None:
    """Pruned entries carry the video, pose, and annotation paths."""
    project = _make_project(mode, ["Walk"], {}, ["a.avi"])

    (result,) = get_videos_to_prune(project)

    assert result == VideoPaths(Path("a.avi"), Path("a.avi_pose.h5"), Path("a.avi.json"))
