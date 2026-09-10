"""Tests for the ``jabs-cli export-video`` command.

The command imports :mod:`jabs.video_export` lazily, inside the function body, so that
registering it does not make the rest of ``jabs-cli`` depend on Qt being importable.
These tests exploit that: they inject a stand-in module into ``sys.modules``, so the
command's argument handling is exercised without Qt and therefore runs everywhere,
including CI runners with no graphics libraries. The real renderer and writer are
covered by ``tests/video_export/``.
"""

import sys
import types
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from jabs.scripts.cli import export_video as ev
from jabs.scripts.cli.cli import cli

_VIDEO_EXPORT = "jabs.video_export"


class _FakeExportError(Exception):
    """Stand-in for ``jabs.video_export.VideoExportError``."""


@pytest.fixture
def export_spy(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Swap in a Qt-free ``jabs.video_export`` and a stub pose loader.

    Returns the spy standing in for ``export_overlay_video``.
    """
    spy = mock.Mock(return_value=7)
    fake = types.ModuleType(_VIDEO_EXPORT)
    fake.DEFAULT_CODEC = "mp4v"
    fake.VideoExportError = _FakeExportError
    fake.export_overlay_video = spy
    monkeypatch.setitem(sys.modules, _VIDEO_EXPORT, fake)
    monkeypatch.setattr(ev, "open_pose_file", lambda path: mock.Mock(name=f"pose:{path}"))
    return spy


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """A stand-in video file with a discoverable pose file beside it."""
    path = tmp_path / "clip.avi"
    path.write_bytes(b"not really a video")
    (tmp_path / "clip_pose_est_v6.h5").write_bytes(b"not really a pose file")
    return path


def _invoke(*args: str):
    return CliRunner().invoke(cli, ["export-video", *args])


def test_default_output_is_beside_the_source(video: Path, export_spy: mock.Mock) -> None:
    """Omitting --output writes <video>_overlay.mp4 next to the video."""
    result = _invoke(str(video))

    assert result.exit_code == 0, result.output
    assert export_spy.call_args.args[1] == video.with_name("clip_overlay.mp4")


def test_pose_file_is_discovered_beside_the_video(
    video: Path, export_spy: mock.Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The highest-version pose file next to the video is used by default.

    Asserts on the path handed to the loader rather than on console output: rich
    wraps long paths at the terminal width, so the filename is not reliably intact
    in what gets printed.
    """
    loader = mock.Mock(return_value=mock.Mock())
    monkeypatch.setattr(ev, "open_pose_file", loader)

    result = _invoke(str(video))

    assert result.exit_code == 0, result.output
    loader.assert_called_once_with(video.with_name("clip_pose_est_v6.h5"))


def test_missing_pose_file_is_a_clean_error(tmp_path: Path, export_spy: mock.Mock) -> None:
    """A video with no pose file fails with a message, not a traceback."""
    lonely = tmp_path / "lonely.avi"
    lonely.write_bytes(b"x")

    result = _invoke(str(lonely))

    assert result.exit_code != 0
    assert not isinstance(result.exception, AttributeError)
    export_spy.assert_not_called()


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--segmentation", True), ("--no-segmentation", False)],
    ids=["on", "off"],
)
def test_segmentation_flag(video: Path, export_spy: mock.Mock, flag: str, expected: bool) -> None:
    """--segmentation/--no-segmentation reaches the exporter."""
    result = _invoke(str(video), flag)

    assert result.exit_code == 0, result.output
    assert export_spy.call_args.kwargs["draw_segmentation"] is expected


def test_segmentation_defaults_on(video: Path, export_spy: mock.Mock) -> None:
    """The default matches Export Frame, which always draws segmentation."""
    _invoke(str(video))

    assert export_spy.call_args.kwargs["draw_segmentation"] is True


def test_codec_defaults_to_the_modules_default(video: Path, export_spy: mock.Mock) -> None:
    """Omitting --codec resolves to DEFAULT_CODEC rather than passing None through."""
    _invoke(str(video))

    assert export_spy.call_args.kwargs["codec"] == "mp4v"


def test_codec_option_is_forwarded(video: Path, export_spy: mock.Mock) -> None:
    """--codec overrides the default fourcc."""
    _invoke(str(video), "--codec", "avc1")

    assert export_spy.call_args.kwargs["codec"] == "avc1"


def test_existing_output_requires_force(video: Path, export_spy: mock.Mock) -> None:
    """An existing output is not clobbered without --force."""
    output = video.with_name("clip_overlay.mp4")
    output.write_bytes(b"previous export")

    result = _invoke(str(video))

    assert result.exit_code != 0
    assert "--force" in result.output
    export_spy.assert_not_called()


def test_force_overwrites_existing_output(video: Path, export_spy: mock.Mock) -> None:
    """--force allows overwriting a previous export."""
    video.with_name("clip_overlay.mp4").write_bytes(b"previous export")

    result = _invoke(str(video), "--force")

    assert result.exit_code == 0, result.output
    export_spy.assert_called_once()


def test_export_error_is_reported_cleanly(video: Path, export_spy: mock.Mock) -> None:
    """A VideoExportError becomes a CLI error message rather than a traceback."""
    export_spy.side_effect = _FakeExportError("codec 'ZZZZ' unavailable")

    result = _invoke(str(video))

    assert result.exit_code != 0
    assert "ZZZZ" in result.output


def test_unimportable_qt_fails_gracefully(video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine without Qt gets an explanation, not an ImportError traceback.

    Rendering the skeleton needs Qt's painter, and Qt needs system graphics libraries
    that headless machines routinely lack. Setting the entry to None makes the import
    raise the same way a missing ``libEGL.so.1`` does.
    """
    monkeypatch.setattr(ev, "open_pose_file", lambda path: mock.Mock())
    monkeypatch.setitem(sys.modules, _VIDEO_EXPORT, None)

    result = _invoke(str(video))

    assert result.exit_code != 0
    # a ClickException is rendered as "Error: ..."; an uncaught ImportError would
    # instead surface as an exception on the result
    assert not isinstance(result.exception, ImportError)
    assert "Error:" in result.output
    assert "Qt" in result.output
    assert "libegl1" in result.output, "should name the package that fixes it"


def test_unimportable_qt_does_not_break_other_subcommands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registering export-video must not make the whole CLI require Qt.

    Regression: importing the command module at registration time pulled in Qt, so
    every other jabs-cli subcommand died with an ImportError on headless machines.
    """
    monkeypatch.setitem(sys.modules, _VIDEO_EXPORT, None)

    assert CliRunner().invoke(cli, ["--help"]).exit_code == 0
    assert CliRunner().invoke(cli, ["export-video", "--help"]).exit_code == 0
    assert CliRunner().invoke(cli, ["compute-features", "--help"]).exit_code == 0


def test_unreadable_pose_file_is_a_clean_error(
    video: Path, export_spy: mock.Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed pose file produces a message, not an h5py traceback."""
    monkeypatch.setattr(
        ev, "open_pose_file", mock.Mock(side_effect=OSError("unable to open file"))
    )

    result = _invoke(str(video))

    assert result.exit_code != 0
    assert "Could not read pose file" in result.output
    export_spy.assert_not_called()
