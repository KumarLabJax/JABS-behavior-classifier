"""Tests for the ``jabs-cli cross-validation`` command option parsing.

These tests exercise the Click command's translation of ``--grouping-strategy`` /
``--grouping-pattern`` into the arguments passed to
:func:`jabs.scripts.cli.cross_validation.run_cross_validation`. The heavy
``run_cross_validation`` implementation is replaced with a spy so the tests stay
fast and do not require a real JABS project on disk.
"""

from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

import jabs.scripts.cli.cli as cli_module
from jabs.classifier import MlflowLoggingError
from jabs.core.enums import CrossValidationGroupingStrategy
from jabs.scripts.cli.cli import cli


@pytest.fixture
def run_cv_spy(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Replace ``run_cross_validation`` (as imported into cli.py) with a spy."""
    spy = mock.Mock()
    monkeypatch.setattr(cli_module, "run_cross_validation", spy)
    return spy


@pytest.fixture
def mlflow_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CLI treat the optional 'mlflow' extra as installed.

    The extra is not a root dependency, so it is typically absent from the test
    environment; tests of the MLflow-enabled path patch this to be deterministic.
    """
    monkeypatch.setattr(cli_module, "mlflow_available", lambda: True)


def _invoke(tmp_path: Path, *extra_args: str):
    """Invoke the cross-validation command against ``tmp_path`` with extra args."""
    runner = CliRunner()
    return runner.invoke(
        cli,
        ["cross-validation", str(tmp_path), "--behavior", "Walk", *extra_args],
    )


@pytest.mark.parametrize(
    ("strategy_arg", "expected"),
    [
        ("video", CrossValidationGroupingStrategy.VIDEO),
        ("individual", CrossValidationGroupingStrategy.INDIVIDUAL),
        ("filename", CrossValidationGroupingStrategy.FILENAME_PATTERN),
        ("FILENAME", CrossValidationGroupingStrategy.FILENAME_PATTERN),
    ],
    ids=["video", "individual", "filename", "filename-uppercase"],
)
def test_grouping_strategy_maps_to_enum(
    tmp_path: Path,
    run_cv_spy: mock.Mock,
    strategy_arg: str,
    expected: CrossValidationGroupingStrategy,
) -> None:
    """``--grouping-strategy`` values (case-insensitive) map to the right enum."""
    result = _invoke(tmp_path, "--grouping-strategy", strategy_arg)

    assert result.exit_code == 0, result.output
    run_cv_spy.assert_called_once()
    # grouping_strategy is the 4th positional argument
    assert run_cv_spy.call_args.args[3] == expected


def test_filename_pattern_passed_as_grouping_regex(tmp_path: Path, run_cv_spy: mock.Mock) -> None:
    """``--grouping-pattern`` is forwarded as the ``grouping_regex`` keyword."""
    result = _invoke(
        tmp_path,
        "--grouping-strategy",
        "filename",
        "--grouping-pattern",
        r"^(\w+?)_",
    )

    assert result.exit_code == 0, result.output
    run_cv_spy.assert_called_once()
    assert run_cv_spy.call_args.args[3] == CrossValidationGroupingStrategy.FILENAME_PATTERN
    assert run_cv_spy.call_args.kwargs["grouping_regex"] == r"^(\w+?)_"


def test_no_strategy_defaults_to_none(tmp_path: Path, run_cv_spy: mock.Mock) -> None:
    """Omitting the strategy/pattern defers to project settings (None passed through)."""
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    run_cv_spy.assert_called_once()
    assert run_cv_spy.call_args.args[3] is None
    assert run_cv_spy.call_args.kwargs["grouping_regex"] is None


def test_invalid_grouping_strategy_rejected(tmp_path: Path, run_cv_spy: mock.Mock) -> None:
    """An unknown strategy is rejected by Click before run_cross_validation is called."""
    result = _invoke(tmp_path, "--grouping-strategy", "bogus")

    assert result.exit_code != 0
    run_cv_spy.assert_not_called()


# --------------------------------------------------------------------------- #
# MLflow options
# --------------------------------------------------------------------------- #
def test_mlflow_absent_disables_logging(tmp_path: Path, run_cv_spy: mock.Mock) -> None:
    """Without --mlflow, logging is disabled and no env file is passed."""
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    kwargs = run_cv_spy.call_args.kwargs
    assert kwargs["mlflow_enabled"] is False
    assert kwargs["mlflow_env_file"] is None
    assert kwargs["mlflow_log_report"] is True


def test_mlflow_bare_flag_enables_ambient(
    tmp_path: Path, run_cv_spy: mock.Mock, mlflow_installed: None
) -> None:
    """A bare --mlflow enables logging with no env file (ambient environment)."""
    result = _invoke(tmp_path, "--mlflow")

    assert result.exit_code == 0, result.output
    kwargs = run_cv_spy.call_args.kwargs
    assert kwargs["mlflow_enabled"] is True
    assert kwargs["mlflow_env_file"] is None


def test_mlflow_with_env_file(
    tmp_path: Path, run_cv_spy: mock.Mock, mlflow_installed: None
) -> None:
    """--mlflow with a path forwards that .env file."""
    result = _invoke(tmp_path, "--mlflow", "settings.env")

    assert result.exit_code == 0, result.output
    kwargs = run_cv_spy.call_args.kwargs
    assert kwargs["mlflow_enabled"] is True
    assert kwargs["mlflow_env_file"] == Path("settings.env")


def test_mlflow_tags_parsed_and_forwarded(
    tmp_path: Path, run_cv_spy: mock.Mock, mlflow_installed: None
) -> None:
    """Repeated --mlflow-tag entries are parsed into a dict."""
    result = _invoke(
        tmp_path, "--mlflow", "--mlflow-tag", "purpose=baseline", "--mlflow-tag", "owner=glen"
    )

    assert result.exit_code == 0, result.output
    assert run_cv_spy.call_args.kwargs["mlflow_tags"] == {
        "purpose": "baseline",
        "owner": "glen",
    }


def test_mlflow_no_report_flag(
    tmp_path: Path, run_cv_spy: mock.Mock, mlflow_installed: None
) -> None:
    """--mlflow-no-report disables the report artifact upload."""
    result = _invoke(tmp_path, "--mlflow", "--mlflow-no-report")

    assert result.exit_code == 0, result.output
    assert run_cv_spy.call_args.kwargs["mlflow_log_report"] is False


def test_mlflow_experiment_forwarded(
    tmp_path: Path, run_cv_spy: mock.Mock, mlflow_installed: None
) -> None:
    """--mlflow-experiment is forwarded; default is None (resolved downstream)."""
    result = _invoke(tmp_path, "--mlflow", "--mlflow-experiment", "my-experiment")
    assert result.exit_code == 0, result.output
    assert run_cv_spy.call_args.kwargs["mlflow_experiment"] == "my-experiment"

    run_cv_spy.reset_mock()
    result = _invoke(tmp_path, "--mlflow")
    assert result.exit_code == 0, result.output
    assert run_cv_spy.call_args.kwargs["mlflow_experiment"] is None


def test_invalid_mlflow_tag_rejected(
    tmp_path: Path, run_cv_spy: mock.Mock, mlflow_installed: None
) -> None:
    """A malformed --mlflow-tag fails before run_cross_validation when MLflow is enabled."""
    result = _invoke(tmp_path, "--mlflow", "--mlflow-tag", "noequals")

    assert result.exit_code != 0
    run_cv_spy.assert_not_called()


def test_mlflow_tag_ignored_without_mlflow(tmp_path: Path, run_cv_spy: mock.Mock) -> None:
    """--mlflow-tag is a no-op (even if malformed) when --mlflow is not given."""
    result = _invoke(tmp_path, "--mlflow-tag", "noequals")

    assert result.exit_code == 0, result.output
    run_cv_spy.assert_called_once()
    assert run_cv_spy.call_args.kwargs["mlflow_enabled"] is False
    assert run_cv_spy.call_args.kwargs["mlflow_tags"] == {}


def test_mlflow_logging_failure_exits_with_code_3(
    tmp_path: Path, run_cv_spy: mock.Mock, mlflow_installed: None
) -> None:
    """An MlflowLoggingError maps to a distinct exit code (3), not the generic 1."""
    run_cv_spy.side_effect = MlflowLoggingError("push failed")

    result = _invoke(tmp_path, "--mlflow")

    assert result.exit_code == 3


def test_mlflow_unavailable_warns_and_ignores(
    tmp_path: Path, run_cv_spy: mock.Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the mlflow extra is absent, --mlflow is ignored with a warning (exit 0)."""
    monkeypatch.setattr(cli_module, "mlflow_available", lambda: False)

    # A malformed tag must NOT error here: the options are ignored when the extra
    # is missing, so the tag is never parsed.
    result = _invoke(tmp_path, "--mlflow", "--mlflow-tag", "noequals")

    assert result.exit_code == 0, result.output
    assert "not installed" in result.stderr
    # cross-validation still runs, but MLflow logging is disabled
    run_cv_spy.assert_called_once()
    assert run_cv_spy.call_args.kwargs["mlflow_enabled"] is False
    assert run_cv_spy.call_args.kwargs["mlflow_tags"] == {}
