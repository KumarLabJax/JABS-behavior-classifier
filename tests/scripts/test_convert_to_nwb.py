"""Tests for convert_to_nwb helper functions."""

import datetime

import pytest

from jabs.scripts.cli.convert_to_nwb import _parse_session_start_time


def test_parse_utc_offset():
    """Test parsing a UTC offset datetime string."""
    dt = _parse_session_start_time("2024-03-15T10:30:00+00:00")
    assert dt == datetime.datetime(2024, 3, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)


def test_parse_negative_offset():
    """Test parsing a negative offset datetime string."""
    dt = _parse_session_start_time("2024-03-15T10:30:00-05:00")
    expected_tz = datetime.timezone(datetime.timedelta(hours=-5))
    assert dt == datetime.datetime(2024, 3, 15, 10, 30, 0, tzinfo=expected_tz)


def test_parse_z_suffix():
    """Test parsing a datetime string with 'Z' suffix (UTC)."""
    dt = _parse_session_start_time("2024-03-15T10:30:00Z")
    assert dt.tzinfo == datetime.timezone.utc
    assert dt.year == 2024 and dt.month == 3 and dt.day == 15


def test_parse_naive_assumes_utc(caplog):
    """Test that naive datetime strings are assumed to be UTC and log a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        dt = _parse_session_start_time("2024-03-15T10:30:00")

    assert dt.tzinfo == datetime.timezone.utc
    assert "no timezone" in caplog.text.lower() or "utc" in caplog.text.lower()


def test_parse_invalid_raises():
    """Test that invalid datetime strings raise ValueError."""
    with pytest.raises(ValueError, match="ISO 8601"):
        _parse_session_start_time("not-a-date")


@pytest.mark.parametrize("value", [42, None, 3.14, True], ids=["int", "null", "float", "bool"])
def test_parse_non_string_raises(value):
    """Test that ValueError is raised if value is not a string."""
    with pytest.raises(ValueError, match="must be a string"):
        _parse_session_start_time(value)
