"""Tests for the SegmentationFeatureGroup class."""

import pytest

from jabs.feature_extraction.segmentation_features import SegmentationFeatureGroup, segment_group


@pytest.fixture
def moment_info_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the identity of every MomentInfo the group builds.

    Returns:
        list that gains one entry (the identity) per MomentInfo construction.
    """
    calls: list[int] = []
    original = segment_group.MomentInfo

    def counting_moment_info(poses, identity, pixel_scale):
        calls.append(identity)
        return original(poses, identity, pixel_scale)

    monkeypatch.setattr(segment_group, "MomentInfo", counting_moment_info)
    return calls


def test_window_after_per_frame_reuses_moment_info(seg_data, moment_info_calls) -> None:
    """Computing per-frame then window features for one identity builds MomentInfo once."""
    group = SegmentationFeatureGroup(seg_data["pose_est_v6"], 1.0)

    per_frame = group.per_frame(0)
    window = group.window(0, 2, per_frame)

    assert moment_info_calls == [0]
    assert set(window) == set(per_frame)


def test_moment_info_is_rebuilt_for_a_different_identity(seg_data, moment_info_calls) -> None:
    """The cached MomentInfo belongs to one identity and is replaced when the identity changes."""
    group = SegmentationFeatureGroup(seg_data["pose_est_v6"], 1.0)

    group._init_feature_mods(0)
    first_cache = group._moments_cache
    group._init_feature_mods(0)
    assert group._moments_cache is first_cache

    group._init_feature_mods(1)
    assert group._moments_cache is not first_cache
    assert moment_info_calls == [0, 1]
