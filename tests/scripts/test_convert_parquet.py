"""Tests for the convert-parquet CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from jabs.scripts.cli.cli import cli


def test_non_parquet_suffix_rejected(tmp_path: Path) -> None:
    """Input file without .parquet suffix produces a clear error."""
    bad_input = tmp_path / "poses.csv"
    bad_input.touch()
    runner = CliRunner()
    result = runner.invoke(cli, ["convert-parquet", str(bad_input)])
    assert result.exit_code != 0
    assert ".parquet" in result.output


def test_convert_called_with_expected_paths(tmp_path: Path) -> None:
    """convert() is invoked with the correct input and output paths."""
    input_file = tmp_path / "session.parquet"
    input_file.touch()
    expected_output = tmp_path / "session_pose_est_v8.h5"

    with patch("jabs.scripts.cli.convert_parquet.convert") as mock_convert:
        runner = CliRunner()
        result = runner.invoke(cli, ["convert-parquet", str(input_file)])

    assert result.exit_code == 0, result.output
    mock_convert.assert_called_once_with(input_file, expected_output, None, 1800)


def test_out_dir_routes_output(tmp_path: Path) -> None:
    """--out-dir places the output file in the specified directory."""
    input_file = tmp_path / "session.parquet"
    input_file.touch()
    out_dir = tmp_path / "output"
    expected_output = out_dir / "session_pose_est_v8.h5"

    with patch("jabs.scripts.cli.convert_parquet.convert") as mock_convert:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["convert-parquet", str(input_file), "--out-dir", str(out_dir)]
        )

    assert result.exit_code == 0, result.output
    mock_convert.assert_called_once_with(input_file, expected_output, None, 1800)
    assert out_dir.exists()


def test_num_frames_forwarded(tmp_path: Path) -> None:
    """--num-frames is passed through to convert()."""
    input_file = tmp_path / "session.parquet"
    input_file.touch()

    with patch("jabs.scripts.cli.convert_parquet.convert") as mock_convert:
        runner = CliRunner()
        result = runner.invoke(cli, ["convert-parquet", str(input_file), "--num-frames", "3600"])

    assert result.exit_code == 0, result.output
    _, _, _, num_frames = mock_convert.call_args.args
    assert num_frames == 3600


def test_lixit_parquet_loaded_and_forwarded(tmp_path: Path) -> None:
    """--lixit-parquet is read and its array passed to convert()."""
    input_file = tmp_path / "session.parquet"
    input_file.touch()
    lixit_file = tmp_path / "lixit.parquet"
    lixit_file.touch()

    fake_lixit = np.zeros((1, 3, 2), dtype=np.float32)

    with (
        patch("jabs.scripts.cli.convert_parquet.read_lixit_parquet") as mock_read,
        patch("jabs.scripts.cli.convert_parquet.convert") as mock_convert,
    ):
        mock_read.return_value = fake_lixit
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["convert-parquet", str(input_file), "--lixit-parquet", str(lixit_file)],
        )

    assert result.exit_code == 0, result.output
    mock_read.assert_called_once_with(lixit_file)
    _, _, lixit_arg, _ = mock_convert.call_args.args
    np.testing.assert_array_equal(lixit_arg, fake_lixit)


def test_lixit_parquet_error_shown_as_cli_error(tmp_path: Path) -> None:
    """A ValueError from read_lixit_parquet is surfaced as a Click error."""
    input_file = tmp_path / "session.parquet"
    input_file.touch()
    lixit_file = tmp_path / "lixit.parquet"
    lixit_file.touch()

    with patch(
        "jabs.scripts.cli.convert_parquet.read_lixit_parquet",
        side_effect=ValueError("missing 'keypoints' column"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["convert-parquet", str(input_file), "--lixit-parquet", str(lixit_file)],
        )

    assert result.exit_code != 0
    assert "keypoints" in result.output


def test_convert_failure_shown_as_cli_error(tmp_path: Path) -> None:
    """An exception from convert() is surfaced as a Click error."""
    input_file = tmp_path / "session.parquet"
    input_file.touch()

    with patch(
        "jabs.scripts.cli.convert_parquet.convert",
        side_effect=RuntimeError("bad data"),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["convert-parquet", str(input_file)])

    assert result.exit_code != 0
    assert "bad data" in result.output
