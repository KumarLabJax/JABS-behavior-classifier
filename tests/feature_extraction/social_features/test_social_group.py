"""Tests for the SocialFeatureGroup class."""

import pytest

from jabs.feature_extraction.social_features import SocialFeatureGroup, social_group


@pytest.fixture
def closest_identity_info_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the identity of every ClosestIdentityInfo the group builds.

    Returns:
        list that gains one entry (the identity) per ClosestIdentityInfo construction.
    """
    calls: list[int] = []
    original = social_group.ClosestIdentityInfo

    def counting_info(poses, identity, pixel_scale):
        calls.append(identity)
        return original(poses, identity, pixel_scale)

    monkeypatch.setattr(social_group, "ClosestIdentityInfo", counting_info)
    return calls


def test_window_after_per_frame_reuses_closest_identity_info(
    pose_est_v5_short, closest_identity_info_calls
) -> None:
    """Computing per-frame then window features for one identity builds the info once."""
    group = SocialFeatureGroup(pose_est_v5_short, 1.0)

    per_frame = group.per_frame(0)
    window = group.window(0, 2, per_frame)

    assert closest_identity_info_calls == [0]
    assert set(window) == set(per_frame)


def test_closest_identity_info_is_rebuilt_for_a_different_identity(
    pose_est_v5_short, closest_identity_info_calls
) -> None:
    """The cached info belongs to one identity and is replaced when the identity changes."""
    assert pose_est_v5_short.num_identities > 1
    group = SocialFeatureGroup(pose_est_v5_short, 1.0)

    group._init_feature_mods(0)
    first_cache = group.closest_identities
    group._init_feature_mods(0)
    assert group.closest_identities is first_cache

    group._init_feature_mods(1)
    assert group.closest_identities is not first_cache
    assert closest_identity_info_calls == [0, 1]
