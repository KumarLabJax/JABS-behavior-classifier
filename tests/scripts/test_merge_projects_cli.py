"""Tests for the ``jabs-cli merge`` subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from jabs.project.project_merge import MergeStrategy
from jabs.scripts.cli.cli import cli


def _make_project_dir(path: Path) -> Path:
    """Create a directory that passes ``Project.is_valid_project_directory``."""
    (path / "jabs").mkdir(parents=True)
    (path / "jabs" / "project.json").write_text("{}")
    return path


def test_merge_strategy_is_required(tmp_path: Path) -> None:
    """Omitting --merge-strategy is an error."""
    dest = _make_project_dir(tmp_path / "dest")
    source = _make_project_dir(tmp_path / "source")

    result = CliRunner().invoke(cli, ["merge", str(dest), str(source)])

    assert result.exit_code != 0
    assert "--merge-strategy" in result.output


def test_same_project_rejected(tmp_path: Path) -> None:
    """Merging a project into itself is rejected before any project is opened."""
    project = _make_project_dir(tmp_path / "project")

    with mock.patch("jabs.scripts.cli.merge_projects.Project") as mock_project:
        result = CliRunner().invoke(
            cli,
            ["merge", str(project), str(project), "--merge-strategy", "destination-wins"],
        )

    assert result.exit_code != 0
    assert "cannot be the same" in result.output
    mock_project.assert_not_called()


def test_invalid_destination_project_rejected(tmp_path: Path) -> None:
    """A destination directory that is not a JABS project is rejected."""
    dest = tmp_path / "dest"
    dest.mkdir()
    source = _make_project_dir(tmp_path / "source")

    result = CliRunner().invoke(
        cli,
        ["merge", str(dest), str(source), "--merge-strategy", "destination-wins"],
    )

    assert result.exit_code != 0
    assert "Destination project" in result.output
    assert "not a valid JABS project" in result.output


def test_invalid_source_project_rejected(tmp_path: Path) -> None:
    """A source directory that is not a JABS project is rejected."""
    dest = _make_project_dir(tmp_path / "dest")
    source = tmp_path / "source"
    source.mkdir()

    result = CliRunner().invoke(
        cli,
        ["merge", str(dest), str(source), "--merge-strategy", "destination-wins"],
    )

    assert result.exit_code != 0
    assert "Source project" in result.output
    assert "not a valid JABS project" in result.output


def test_missing_project_directory_rejected(tmp_path: Path) -> None:
    """A nonexistent project path fails click's path validation."""
    dest = _make_project_dir(tmp_path / "dest")
    missing = tmp_path / "does_not_exist"

    result = CliRunner().invoke(
        cli,
        ["merge", str(dest), str(missing), "--merge-strategy", "destination-wins"],
    )

    assert result.exit_code != 0
    assert "does_not_exist" in result.output


def test_merge_invoked_with_selected_strategy(tmp_path: Path) -> None:
    """Valid inputs open both projects and forward the selected merge strategy."""
    dest = _make_project_dir(tmp_path / "dest")
    source = _make_project_dir(tmp_path / "source")

    # distinct instances so the assertion below fails if the same project is
    # forwarded twice, or if only one of the two is ever instantiated
    dest_project = mock.Mock(name="destination_project")
    source_project = mock.Mock(name="source_project")

    with (
        mock.patch(
            "jabs.scripts.cli.merge_projects.Project",
            side_effect=[dest_project, source_project],
        ) as mock_project,
        mock.patch("jabs.scripts.cli.merge_projects.merge_projects") as mock_merge,
    ):
        result = CliRunner().invoke(
            cli,
            ["merge", str(dest), str(source), "--merge-strategy", "behavior-wins"],
        )

    assert result.exit_code == 0, result.output
    assert mock_project.call_args_list == [mock.call(dest), mock.call(source)]
    mock_merge.assert_called_once_with(
        dest_project,
        source_project,
        MergeStrategy.BEHAVIOR_WINS,
    )
