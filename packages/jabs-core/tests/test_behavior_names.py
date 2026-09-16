"""Unit tests for jabs.core.utils.behavior_names module."""

import pytest

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.utils import validate_behavior_names


@pytest.mark.parametrize(
    "behavior_names",
    [[], ["walk"], ["walk", "groom", "rear"]],
    ids=["empty", "single", "several"],
)
def test_valid_names_accepted(behavior_names: list[str]) -> None:
    """A list free of the reserved name and of duplicates is accepted."""
    assert validate_behavior_names(behavior_names) is None


def test_reserved_name_raises() -> None:
    """The reserved multi-class class name may not appear in the behavior list."""
    with pytest.raises(ValueError, match="reserved name"):
        validate_behavior_names(["walk", MULTICLASS_NONE_BEHAVIOR])


def test_duplicate_names_raises() -> None:
    """A repeated behavior name is rejected."""
    with pytest.raises(ValueError, match="duplicate"):
        validate_behavior_names(["walk", "groom", "walk"])


def test_reserved_name_checked_before_duplicates() -> None:
    """A list that is both duplicated and reserved reports the reserved name."""
    with pytest.raises(ValueError, match="reserved name"):
        validate_behavior_names([MULTICLASS_NONE_BEHAVIOR, MULTICLASS_NONE_BEHAVIOR])


def test_accepts_any_sequence() -> None:
    """Callers may pass a tuple as well as a list."""
    validate_behavior_names(("walk", "groom"))
    with pytest.raises(ValueError, match="duplicate"):
        validate_behavior_names(("walk", "walk"))
