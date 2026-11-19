"""The ``jabs-pose-file`` command converts and validates, and exits with a meaningful code."""

import importlib.metadata
import shutil
from pathlib import Path

import pytest

from jabs.io.internal.pose_file.cli import main
from jabs.io.internal.pose_file.validate import validate

from .conftest import legacy

POSE_FORMAT = Path(__file__).parents[3] / "data" / "pose-format"


def _convert(source: Path, out: Path) -> int:
    """Run ``convert`` with the video facts the legacy file does not record."""
    return main(
        ["convert", str(source), str(out), "--width", "800", "--height", "800", "--fps", "30"]
    )


def test_convert_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``convert`` writes a valid file, reports it on stdout and exits 0."""
    out = tmp_path / "out.h5"

    assert _convert(legacy("v8-edge-cases.h5"), out) == 0

    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []
    captured = capsys.readouterr()
    assert str(out) in captured.out
    assert captured.err == ""


def test_convert_block_mb(tmp_path: Path) -> None:
    """``--block-mb`` is accepted and the result is still valid."""
    out = tmp_path / "out.h5"

    assert main(["convert", str(legacy("v8-with-centers.h5")), str(out), "--block-mb", "1"]) == 0
    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []


def test_convert_refusal_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A refusal prints one ``error:`` line naming the frame, with no traceback."""
    out = tmp_path / "out.h5"

    assert _convert(legacy("v8-duplicate-embed.h5"), out) == 1

    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "frame 100" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not out.exists()


def test_convert_missing_source_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A source that cannot be opened is an ``error:`` line, not a traceback."""
    assert _convert(tmp_path / "absent.h5", tmp_path / "out.h5") == 1

    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_convert_onto_the_source_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Converting a file onto itself is refused with an ``error:`` line, leaving it as it was."""
    src = tmp_path / "v8-with-centers.h5"
    shutil.copyfile(legacy("v8-with-centers.h5"), src)
    before = src.read_bytes()

    assert _convert(src, src) == 1

    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "is the source" in captured.err
    assert src.read_bytes() == before


@pytest.mark.parametrize(
    "extra",
    [
        ["--block-mb", "0"],
        ["--block-mb", "1.5"],
        ["--width", "-800"],
        ["--height", "0"],
        ["--fps", "0"],
        ["--fps", "nan"],
        ["--fps", "thirty"],
    ],
)
def test_convert_usage_errors_exit_2(tmp_path: Path, extra: list[str]) -> None:
    """A malformed option is a usage error: argparse exits 2 before anything is read."""
    with pytest.raises(SystemExit) as raised:
        main(["convert", str(legacy("v8-edge-cases.h5")), str(tmp_path / "out.h5"), *extra])
    assert raised.value.code == 2
    assert not (tmp_path / "out.h5").exists()


def test_missing_subcommand_exits_2() -> None:
    """The command needs a subcommand."""
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_validate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``validate`` exits 0 on a converted file and 1 on one with an error, naming it."""
    out = tmp_path / "out.h5"
    assert _convert(legacy("v8-edge-cases.h5"), out) == 0
    capsys.readouterr()

    assert main(["validate", str(out), "--deep"]) == 0
    assert "error" not in capsys.readouterr().out

    assert main(["validate", str(POSE_FORMAT / "invalid-mask-dtype.h5")]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert any("error mask_reference:" in line for line in lines)


@pytest.mark.parametrize("name", ["invalid-not-hdf5.h5", "invalid-not-a-pose-file.h5"])
def test_validate_non_pose_file_exits_1(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    """A file that is not HDF5, or not a pose file, prints its findings and exits 1."""
    assert main(["validate", str(POSE_FORMAT / name)]) == 1

    captured = capsys.readouterr()
    assert captured.out.startswith("error ")
    assert "Traceback" not in captured.err


def test_console_script_is_declared() -> None:
    """The package declares ``jabs-pose-file`` as a console script."""
    try:
        importlib.metadata.distribution("jabs-io")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("jabs-io is not installed with metadata")

    scripts = importlib.metadata.entry_points(group="console_scripts")
    matches = [ep for ep in scripts if ep.name == "jabs-pose-file"]
    assert [ep.value for ep in matches] == ["jabs.io.internal.pose_file.cli:main"]
