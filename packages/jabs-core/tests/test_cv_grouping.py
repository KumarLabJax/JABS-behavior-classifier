"""Tests for cross-validation grouping enum and filename-pattern helpers."""

import pytest

from jabs.core.enums import (
    CrossValidationGroupingStrategy,
    compile_grouping_regex,
    filename_group_key,
)


def test_filename_pattern_member_exists() -> None:
    """The FILENAME_PATTERN strategy is available with its display value."""
    assert CrossValidationGroupingStrategy.FILENAME_PATTERN.value == "Filename Pattern"
    assert CrossValidationGroupingStrategy("Filename Pattern") is (
        CrossValidationGroupingStrategy.FILENAME_PATTERN
    )


def test_compile_grouping_regex_valid() -> None:
    """A valid pattern compiles to a usable regex."""
    pattern = compile_grouping_regex(r"cage_(\d+)")
    assert pattern.search("cage_0042.mp4") is not None


@pytest.mark.parametrize("regex", ["", None], ids=["empty", "none"])
def test_compile_grouping_regex_empty_raises(regex) -> None:
    """An empty (or falsy) pattern is rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        compile_grouping_regex(regex)


def test_compile_grouping_regex_invalid_raises() -> None:
    """A syntactically invalid pattern raises ValueError, not re.error."""
    with pytest.raises(ValueError, match="Invalid filename grouping pattern"):
        compile_grouping_regex("cage_(")


def test_filename_group_key_uses_capture_group() -> None:
    """When the pattern has a capture group, the captured text is the key."""
    pattern = compile_grouping_regex(r"cage_(\d+)")
    assert filename_group_key("cage_0042_2026-06-16.mp4", pattern) == "0042"


def test_filename_group_key_uses_full_match_without_capture_group() -> None:
    """Without a capture group, the whole matched substring is the key."""
    pattern = compile_grouping_regex(r"cage_\d+")
    assert filename_group_key("cage_0042_2026-06-16.mp4", pattern) == "cage_0042"


def test_filename_group_key_searches_anywhere() -> None:
    """The pattern matches anywhere in the filename (re.search semantics)."""
    pattern = compile_grouping_regex(r"cage_(\d+)")
    assert filename_group_key("2026-06-16_cage_0007_cam1.avi", pattern) == "0007"


def test_filename_group_key_unmatched_returns_filename() -> None:
    """A filename that does not match becomes its own group (keyed by the name)."""
    pattern = compile_grouping_regex(r"cage_(\d+)")
    assert filename_group_key("mouse_video.mp4", pattern) == "mouse_video.mp4"


def test_filename_group_key_same_cage_different_files_share_key() -> None:
    """Different files from the same cage produce the same grouping key."""
    pattern = compile_grouping_regex(r"cage_(\d+)")
    key_a = filename_group_key("cage_0042_day1.mp4", pattern)
    key_b = filename_group_key("cage_0042_day2.avi", pattern)
    assert key_a == key_b == "0042"


def test_filename_group_key_optional_capture_group_falls_back_to_full_match() -> None:
    """An optional capture group that does not participate falls back to the full match."""
    pattern = compile_grouping_regex(r"cage(_extra)?_\d+")
    # The optional group does not match here, so the full match is used.
    assert filename_group_key("cage_0042.mp4", pattern) == "cage_0042"
