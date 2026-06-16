"""Tests for Project._assign_cv_group_ids cross-validation group assignment."""

import pytest

from jabs.core.enums import CrossValidationGroupingStrategy
from jabs.project.project import Project


def test_assign_cv_group_ids_filename_pattern_groups_videos_by_key() -> None:
    """Videos whose filenames share a regex key are merged into one CV group."""
    videos = ["cage_1_a.mp4", "cage_1_b.mp4", "cage_2_a.mp4", "loose.mp4"]
    all_group_keys = [
        ("cage_1_a.mp4", 0),
        ("cage_1_a.mp4", 1),
        ("cage_1_b.mp4", 0),
        ("cage_2_a.mp4", 0),
        ("loose.mp4", 0),
    ]

    key_to_gid, group_mapping = Project._assign_cv_group_ids(
        all_group_keys,
        videos,
        CrossValidationGroupingStrategy.FILENAME_PATTERN,
        regex=r"cage_(\d+)",
    )

    # All identities of cage 1 (across both videos) share one group id.
    cage1_gid = key_to_gid[("cage_1_a.mp4", 0)]
    assert key_to_gid[("cage_1_a.mp4", 1)] == cage1_gid
    assert key_to_gid[("cage_1_b.mp4", 0)] == cage1_gid

    # Cage 2 and the unmatched video are each their own group.
    cage2_gid = key_to_gid[("cage_2_a.mp4", 0)]
    loose_gid = key_to_gid[("loose.mp4", 0)]
    assert len({cage1_gid, cage2_gid, loose_gid}) == 3

    assert group_mapping[cage1_gid]["label"] == "1"
    assert group_mapping[cage1_gid]["videos"] == ["cage_1_a.mp4", "cage_1_b.mp4"]
    assert group_mapping[cage1_gid]["video"] is None
    assert group_mapping[cage1_gid]["identity"] is None

    # An unmatched filename forms its own group keyed by the filename itself.
    assert group_mapping[loose_gid]["label"] == "loose.mp4"
    assert group_mapping[loose_gid]["videos"] == ["loose.mp4"]


def test_assign_cv_group_ids_filename_pattern_ids_are_contiguous() -> None:
    """Group ids are assigned contiguously from zero in row order."""
    videos = ["cage_2_x.mp4", "cage_1_y.mp4"]
    all_group_keys = [("cage_2_x.mp4", 0), ("cage_1_y.mp4", 0)]

    key_to_gid, group_mapping = Project._assign_cv_group_ids(
        all_group_keys,
        videos,
        CrossValidationGroupingStrategy.FILENAME_PATTERN,
        regex=r"cage_(\d+)",
    )

    assert sorted(group_mapping) == [0, 1]
    # First key encountered ("cage_2_x") gets gid 0.
    assert key_to_gid[("cage_2_x.mp4", 0)] == 0
    assert group_mapping[0]["label"] == "2"


def test_assign_cv_group_ids_filename_pattern_requires_regex() -> None:
    """Filename-pattern grouping with an empty regex raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        Project._assign_cv_group_ids(
            [("cage_1_a.mp4", 0)],
            ["cage_1_a.mp4"],
            CrossValidationGroupingStrategy.FILENAME_PATTERN,
            regex="",
        )


def test_assign_cv_group_ids_video_grouping_unchanged() -> None:
    """Regression: VIDEO grouping still assigns one group per video."""
    videos = ["video_a.mp4", "video_b.mp4"]
    all_group_keys = [("video_a.mp4", 0), ("video_a.mp4", 1), ("video_b.mp4", 0)]

    key_to_gid, group_mapping = Project._assign_cv_group_ids(
        all_group_keys, videos, CrossValidationGroupingStrategy.VIDEO
    )

    assert key_to_gid[("video_a.mp4", 0)] == key_to_gid[("video_a.mp4", 1)]
    assert key_to_gid[("video_a.mp4", 0)] != key_to_gid[("video_b.mp4", 0)]
    assert group_mapping[key_to_gid[("video_a.mp4", 0)]] == {
        "video": "video_a.mp4",
        "identity": None,
    }


def test_assign_cv_group_ids_individual_grouping_unchanged() -> None:
    """Regression: INDIVIDUAL grouping assigns one group per (video, identity)."""
    videos = ["video_a.mp4"]
    all_group_keys = [("video_a.mp4", 0), ("video_a.mp4", 1)]

    key_to_gid, group_mapping = Project._assign_cv_group_ids(
        all_group_keys, videos, CrossValidationGroupingStrategy.INDIVIDUAL
    )

    assert key_to_gid[("video_a.mp4", 0)] != key_to_gid[("video_a.mp4", 1)]
    assert group_mapping[key_to_gid[("video_a.mp4", 1)]] == {
        "video": "video_a.mp4",
        "identity": 1,
    }
