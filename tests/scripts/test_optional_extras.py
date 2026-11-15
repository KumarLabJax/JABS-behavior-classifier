"""Tests for the install advice CLI commands give when an optional extra is missing."""

from unittest import mock

from click.testing import CliRunner

from jabs.core.exceptions import MissingOptionalDependencyError
from jabs.scripts.cli import link_nwb_video
from jabs.scripts.cli.cli import cli
from jabs.scripts.cli.optional_extras import missing_extra_error

_LIBRARY_ADVICE = "Install with: pip install 'jabs-io[nwb]'"


def _missing_nwb() -> MissingOptionalDependencyError:
    """The error jabs-io raises when the NWB packages are not installed."""
    return MissingOptionalDependencyError(f"pynwb is required. {_LIBRARY_ADVICE}", extra="nwb")


def test_missing_extra_error_names_the_jabs_extra() -> None:
    """The message tells JABS users to install JABS's extra, not the library's."""
    message = missing_extra_error(_missing_nwb()).message

    assert "'nwb' extra" in message
    assert 'pip install "jabs-behavior-classifier[nwb]"' in message
    assert "Optional Extras" in message
    assert "jabs-io" not in message


def test_convert_to_nwb_gives_jabs_install_advice(monkeypatch, tmp_path) -> None:
    """convert-to-nwb replaces the library's advice with the JABS extra."""
    monkeypatch.setattr(
        "jabs.scripts.cli.cli.run_conversion", mock.Mock(side_effect=_missing_nwb())
    )
    input_path = tmp_path / "session_pose_est_v6.h5"
    input_path.write_bytes(b"")  # must exist for click.Path(exists=True)

    result = CliRunner().invoke(
        cli, ["convert-to-nwb", str(input_path), str(tmp_path / "session.nwb")]
    )

    assert result.exit_code != 0
    assert "jabs-behavior-classifier[nwb]" in result.output
    assert "jabs-io" not in result.output


def test_link_nwb_video_gives_jabs_install_advice(monkeypatch, tmp_path) -> None:
    """link-nwb-video replaces the library's advice with the JABS extra."""
    monkeypatch.setattr(
        link_nwb_video, "require_video_link_support", mock.Mock(side_effect=_missing_nwb())
    )
    (tmp_path / "session.nwb").write_bytes(b"")  # only needs to be found, never read

    result = CliRunner().invoke(cli, ["link-nwb-video", str(tmp_path), str(tmp_path)])

    assert result.exit_code != 0
    assert "jabs-behavior-classifier[nwb]" in result.output
    assert "jabs-io" not in result.output


def test_link_nwb_video_passes_other_import_errors_through(monkeypatch, tmp_path) -> None:
    """An ImportError that is not a missing extra (e.g. pynwb too old) is shown as-is."""
    monkeypatch.setattr(
        link_nwb_video,
        "require_video_link_support",
        mock.Mock(side_effect=ImportError("requires pynwb>=4.0")),
    )
    (tmp_path / "session.nwb").write_bytes(b"")

    result = CliRunner().invoke(cli, ["link-nwb-video", str(tmp_path), str(tmp_path)])

    assert result.exit_code != 0
    assert "requires pynwb>=4.0" in result.output
    assert "jabs-behavior-classifier[nwb]" not in result.output
