"""Tests for resolve_identity_subjects.

Deliberately not guarded by ``pytest.importorskip``: the function shares the NWB
adapter's naming and lookup logic but touches no NWB types, so it must keep
working without the ``[nwb]`` extra installed.
"""

import numpy as np
import pytest

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types import PoseData
from jabs.io.internal.pose import IdentitySubject, resolve_identity_subjects


def _pose_data(num_identities=2, external_ids=None, subjects=None):
    """Build a minimal PoseData with the given identities and subject metadata."""
    body_parts = [kpt.name for kpt in PoseEstimation.KeypointIndex]
    num_keypoints = len(body_parts)
    return PoseData(
        points=np.zeros((num_identities, 3, num_keypoints, 2)),
        point_mask=np.ones((num_identities, 3, num_keypoints), dtype=bool),
        identity_mask=np.ones((num_identities, 3), dtype=bool),
        body_parts=body_parts,
        edges=[],
        fps=30,
        external_ids=external_ids,
        subjects=subjects,
    )


def test_returns_one_entry_per_identity_in_order() -> None:
    """One entry per identity, ordered by identity index."""
    resolved = resolve_identity_subjects(_pose_data(num_identities=3))

    assert [entry.identity_name for entry in resolved] == [
        "subject_1",
        "subject_2",
        "subject_3",
    ]


def test_entries_are_identity_subject_named_tuples() -> None:
    """Entries are IdentitySubject tuples with the expected fields."""
    (entry,) = resolve_identity_subjects(_pose_data(num_identities=1))

    assert isinstance(entry, IdentitySubject)
    assert entry.identity_name == "subject_1"
    assert entry.lookup_keys == ("subject_1",)
    assert entry.metadata == {}


def test_fallback_names_are_one_based() -> None:
    """Names are subject_1..subject_N; a 0-based --subjects file matches nothing."""
    resolved = resolve_identity_subjects(_pose_data(num_identities=2))

    assert "subject_0" not in [entry.identity_name for entry in resolved]


def test_external_ids_are_sanitized_for_the_container_name() -> None:
    """Container names are sanitized external IDs."""
    resolved = resolve_identity_subjects(
        _pose_data(num_identities=2, external_ids=["mouse/a", "mouse b"])
    )

    assert [entry.identity_name for entry in resolved] == ["mouse_a", "mouse_b"]


def test_lookup_keys_include_raw_and_sanitized_when_they_differ() -> None:
    """Both lookup keys are offered when sanitizing changes the name."""
    (entry,) = resolve_identity_subjects(_pose_data(num_identities=1, external_ids=["mouse/a"]))

    assert entry.lookup_keys == ("mouse/a", "mouse_a")


def test_lookup_keys_collapse_when_sanitizing_is_a_no_op() -> None:
    """A single lookup key is offered when sanitizing changes nothing."""
    (entry,) = resolve_identity_subjects(_pose_data(num_identities=1, external_ids=["mouse_a"]))

    assert entry.lookup_keys == ("mouse_a",)


def test_metadata_resolves_by_raw_external_id() -> None:
    """Metadata is found under the raw external ID."""
    (entry,) = resolve_identity_subjects(
        _pose_data(num_identities=1, external_ids=["mouse/a"], subjects={"mouse/a": {"sex": "M"}})
    )

    assert entry.metadata == {"sex": "M"}


def test_metadata_falls_back_to_the_sanitized_name() -> None:
    """Metadata is found under the sanitized container name."""
    (entry,) = resolve_identity_subjects(
        _pose_data(num_identities=1, external_ids=["mouse/a"], subjects={"mouse_a": {"sex": "F"}})
    )

    assert entry.metadata == {"sex": "F"}


def test_raw_key_wins_over_the_sanitized_name() -> None:
    """The raw external ID takes precedence when both keys exist."""
    (entry,) = resolve_identity_subjects(
        _pose_data(
            num_identities=1,
            external_ids=["mouse/a"],
            subjects={"mouse/a": {"sex": "M"}, "mouse_a": {"sex": "F"}},
        )
    )

    assert entry.metadata == {"sex": "M"}


def test_unmatched_identity_gets_empty_metadata() -> None:
    """An identity with no matching key resolves to an empty dict."""
    resolved = resolve_identity_subjects(
        _pose_data(num_identities=2, subjects={"subject_1": {"sex": "M"}})
    )

    assert resolved[0].metadata == {"sex": "M"}
    assert resolved[1].metadata == {}


def test_no_subjects_gives_every_identity_empty_metadata() -> None:
    """With subjects=None every identity resolves to an empty dict."""
    resolved = resolve_identity_subjects(_pose_data(num_identities=2, subjects=None))

    assert all(entry.metadata == {} for entry in resolved)


@pytest.mark.parametrize("num_identities", [1, 2, 5], ids=["one", "two", "five"])
def test_entry_count_matches_identity_count(num_identities: int) -> None:
    """The number of entries tracks the identity count."""
    resolved = resolve_identity_subjects(_pose_data(num_identities=num_identities))

    assert len(resolved) == num_identities


def test_matched_key_is_the_one_the_writer_reads() -> None:
    """With both keys present the raw external ID wins, so the other is shadowed."""
    (entry,) = resolve_identity_subjects(
        _pose_data(
            num_identities=1,
            external_ids=["mouse/a"],
            subjects={"mouse/a": {"sex": "M"}, "mouse_a": {"sex": "F"}},
        )
    )

    assert entry.matched_key == "mouse/a"
    assert entry.lookup_keys == ("mouse/a", "mouse_a")


def test_matched_key_falls_back_to_the_sanitized_name() -> None:
    """Only the sanitized key is present, so that is what the writer reads."""
    (entry,) = resolve_identity_subjects(
        _pose_data(num_identities=1, external_ids=["mouse/a"], subjects={"mouse_a": {"sex": "F"}})
    )

    assert entry.matched_key == "mouse_a"


def test_matched_key_is_none_without_metadata() -> None:
    """An identity with no entry reports no matched key."""
    resolved = resolve_identity_subjects(
        _pose_data(num_identities=2, subjects={"subject_1": {"sex": "M"}})
    )

    assert resolved[0].matched_key == "subject_1"
    assert resolved[1].matched_key is None
