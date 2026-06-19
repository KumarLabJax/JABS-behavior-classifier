"""Tests for convert_to_nwb helper functions."""

import datetime
from unittest import mock

import pytest
from click.testing import CliRunner

from jabs.scripts.cli.convert_to_nwb import _parse_session_start_time, run_conversion


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


# ---------------------------------------------------------------------------
# run_conversion write-mode wiring
# ---------------------------------------------------------------------------


def _patch_conversion_internals(monkeypatch):
    """Patch the pose-loading and save boundaries of run_conversion; return the save mock."""
    pose = mock.Mock(num_identities=2, num_frames=10, fps=30)
    monkeypatch.setattr("jabs.scripts.cli.convert_to_nwb.open_pose_file", lambda *a, **k: pose)
    monkeypatch.setattr(
        "jabs.scripts.cli.convert_to_nwb.pose_to_pose_data",
        lambda *a, **k: mock.sentinel.pose_data,
    )
    save_mock = mock.Mock()
    monkeypatch.setattr("jabs.scripts.cli.convert_to_nwb.save", save_mock)
    return save_mock


def test_run_conversion_multisubject_forwarded(monkeypatch, tmp_path):
    """run_conversion(multisubject=True) forwards multisubject=True to save()."""
    save_mock = _patch_conversion_internals(monkeypatch)

    run_conversion(tmp_path / "in_pose_est_v6.h5", tmp_path / "out.nwb", multisubject=True)

    save_mock.assert_called_once()
    assert save_mock.call_args.kwargs["multisubject"] is True


def test_run_conversion_defaults_to_per_identity(monkeypatch, tmp_path):
    """run_conversion defaults to multisubject=False (per-identity output)."""
    save_mock = _patch_conversion_internals(monkeypatch)

    run_conversion(tmp_path / "in_pose_est_v6.h5", tmp_path / "out.nwb")

    assert save_mock.call_args.kwargs["multisubject"] is False


# ---------------------------------------------------------------------------
# convert-to-nwb CLI wiring
# ---------------------------------------------------------------------------


def test_cli_multisubject_flag_forwarded(monkeypatch, tmp_path):
    """The --multisubject flag is forwarded to run_conversion."""
    from jabs.scripts.cli.cli import cli

    run_mock = mock.Mock()
    monkeypatch.setattr("jabs.scripts.cli.cli.run_conversion", run_mock)
    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")  # must exist for click.Path(exists=True)
    output = tmp_path / "session.nwb"

    result = CliRunner().invoke(
        cli, ["convert-to-nwb", str(input_path), str(output), "--multisubject"]
    )

    assert result.exit_code == 0, result.output
    assert run_mock.call_args.kwargs["multisubject"] is True


def test_cli_defaults_to_per_identity(monkeypatch, tmp_path):
    """Without --multisubject the CLI requests per-identity output (multisubject=False)."""
    from jabs.scripts.cli.cli import cli

    run_mock = mock.Mock()
    monkeypatch.setattr("jabs.scripts.cli.cli.run_conversion", run_mock)
    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")
    output = tmp_path / "session.nwb"

    result = CliRunner().invoke(cli, ["convert-to-nwb", str(input_path), str(output)])

    assert result.exit_code == 0, result.output
    assert run_mock.call_args.kwargs["multisubject"] is False


def test_cli_per_identity_flag_removed(tmp_path):
    """The old --per-identity flag no longer exists."""
    from jabs.scripts.cli.cli import cli

    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")
    output = tmp_path / "session.nwb"

    result = CliRunner().invoke(
        cli, ["convert-to-nwb", str(input_path), str(output), "--per-identity"]
    )

    assert result.exit_code != 0
    assert "no such option" in result.output.lower()
