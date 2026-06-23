"""Tests for :mod:`jabs.classifier.mlflow_logging`.

The actual MLflow client is never imported here; tests that exercise
:func:`log_cross_validation_to_mlflow` inject a fake ``mlflow`` module into
``sys.modules`` so no tracking server, filesystem, or network is touched.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jabs.classifier import mlflow_logging
from jabs.classifier.mlflow_logging import (
    MlflowLoggingError,
    aggregate_cv_metrics,
    build_params,
    build_tags,
    load_env_file,
    log_cross_validation_to_mlflow,
    mlflow_available,
    parse_kv_tags,
)
from jabs.classifier.training_report import BinaryCVResult, TrainingReportData
from jabs.core.enums import CrossValidationGroupingStrategy


@pytest.fixture
def binary_report() -> TrainingReportData:
    """A binary cross-validation report with two iterations."""
    cv_results = [
        BinaryCVResult(
            iteration=1,
            test_label="cage_1",
            accuracy=0.9,
            confusion_matrix=np.zeros((2, 2)),
            precision_behavior=0.8,
            recall_behavior=0.7,
            f1_behavior=0.75,
        ),
        BinaryCVResult(
            iteration=2,
            test_label="cage_2",
            accuracy=0.8,
            confusion_matrix=np.zeros((2, 2)),
            precision_behavior=0.6,
            recall_behavior=0.5,
            f1_behavior=0.55,
        ),
    ]
    return TrainingReportData(
        behavior_name="Walk",
        classifier_type="XGBoost",
        window_size=5,
        balance_training_labels=True,
        symmetric_behavior=False,
        distance_unit="cm",
        cv_results=cv_results,
        final_top_features=[("feat_a", 0.5)],
        training_time_ms=1234,
        timestamp=datetime(2026, 6, 23, 12, 0, 0),
        cv_grouping_strategy=CrossValidationGroupingStrategy.FILENAME_PATTERN,
        frames_behavior=100,
        frames_not_behavior=200,
        bouts_behavior=10,
        bouts_not_behavior=20,
        cv_grouping_regex=r"^(\w+?)_",
    )


class _FakeRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id="run-123")

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeMlflow:
    """Minimal stand-in recording the calls the logger makes."""

    def __init__(self) -> None:
        self.run_name: str | None = None
        self.metrics: dict[str, float] = {}
        self.params: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[str] = []

    def start_run(self, run_name: str | None = None) -> _FakeRun:
        self.run_name = run_name
        return _FakeRun()

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value

    def log_params(self, params: dict[str, str]) -> None:
        self.params.update(params)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def get_tracking_uri(self) -> str:
        return "file:///tmp/mlruns"


# --------------------------------------------------------------------------- #
# mlflow_available
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("spec", "expected"), [(object(), True), (None, False)], ids=["present", "absent"]
)
def test_mlflow_available(spec: object, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """mlflow_available() reflects whether find_spec locates the package."""
    monkeypatch.setattr(mlflow_logging.importlib.util, "find_spec", lambda name: spec)
    assert mlflow_available() is expected


# --------------------------------------------------------------------------- #
# parse_kv_tags
# --------------------------------------------------------------------------- #
def test_parse_kv_tags_basic() -> None:
    """KEY=VALUE entries parse into a dict, values may contain spaces."""
    assert parse_kv_tags(["a=1", "purpose=release candidate"]) == {
        "a": "1",
        "purpose": "release candidate",
    }


def test_parse_kv_tags_none_and_empty() -> None:
    """None or empty input yields an empty dict."""
    assert parse_kv_tags(None) == {}
    assert parse_kv_tags([]) == {}


def test_parse_kv_tags_value_may_contain_equals() -> None:
    """Only the first '=' splits; later ones go to the value."""
    assert parse_kv_tags(["expr=a=b"]) == {"expr": "a=b"}


@pytest.mark.parametrize("bad", ["noequals", "=novalue"], ids=["no-eq", "empty-key"])
def test_parse_kv_tags_invalid(bad: str) -> None:
    """Entries with no '=' or an empty key are rejected."""
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        parse_kv_tags([bad])


# --------------------------------------------------------------------------- #
# load_env_file
# --------------------------------------------------------------------------- #
def test_load_env_file_none_is_noop() -> None:
    """A None env file applies nothing and returns an empty dict."""
    assert load_env_file(None) == {}


def test_load_env_file_applies_only_mlflow_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only MLFLOW_* keys are applied to the environment; others are ignored."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    env_file = tmp_path / "mlflow.env"
    env_file.write_text(
        "# a comment\n"
        'MLFLOW_TRACKING_URI="https://mlflow.example.org"\n'
        "export MLFLOW_EXPERIMENT_NAME=behaviors\n"
        "OTHER_VAR=should-be-ignored\n"
    )

    applied = load_env_file(env_file)

    assert applied == {
        "MLFLOW_TRACKING_URI": "https://mlflow.example.org",
        "MLFLOW_EXPERIMENT_NAME": "behaviors",
    }
    assert os.environ["MLFLOW_TRACKING_URI"] == "https://mlflow.example.org"
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "behaviors"
    assert "OTHER_VAR" not in os.environ


def test_load_env_file_missing_raises(tmp_path: Path) -> None:
    """A given-but-missing env file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="env file not found"):
        load_env_file(tmp_path / "does_not_exist.env")


# --------------------------------------------------------------------------- #
# aggregate_cv_metrics / build_params / build_tags
# --------------------------------------------------------------------------- #
def test_aggregate_cv_metrics_binary(binary_report: TrainingReportData) -> None:
    """Binary CV results aggregate into mean/std and composition metrics."""
    metrics = aggregate_cv_metrics(binary_report)
    assert metrics["cv_accuracy_mean"] == pytest.approx(0.85)
    assert metrics["cv_accuracy_std"] == pytest.approx(0.05)
    assert metrics["cv_iterations"] == pytest.approx(2.0)
    assert metrics["cv_precision_behavior_mean"] == pytest.approx(0.7)
    assert metrics["cv_recall_behavior_mean"] == pytest.approx(0.6)
    assert metrics["cv_f1_behavior_mean"] == pytest.approx(0.65)
    assert metrics["frames_behavior"] == pytest.approx(100.0)
    assert metrics["bouts_not_behavior"] == pytest.approx(20.0)
    assert metrics["training_time_ms"] == pytest.approx(1234.0)


def test_aggregate_cv_metrics_empty(binary_report: TrainingReportData) -> None:
    """No CV iterations yields no metrics."""
    binary_report.cv_results = []
    assert aggregate_cv_metrics(binary_report) == {}


def test_build_params_includes_regex_for_filename_strategy(
    binary_report: TrainingReportData,
) -> None:
    """The grouping regex is recorded as a param under the filename strategy."""
    params = build_params(binary_report)
    assert params["behavior"] == "Walk"
    assert params["classifier"] == "XGBoost"
    assert params["window_size"] == "5"
    assert params["balance_labels"] == "True"
    assert params["cv_grouping_strategy"] == "Filename Pattern"
    assert params["cv_grouping_regex"] == r"^(\w+?)_"


def test_build_params_omits_regex_when_unset(binary_report: TrainingReportData) -> None:
    """No regex param is recorded when no grouping regex is set."""
    binary_report.cv_grouping_strategy = CrossValidationGroupingStrategy.VIDEO
    binary_report.cv_grouping_regex = None
    params = build_params(binary_report)
    assert "cv_grouping_regex" not in params


def test_build_tags_omits_empty(
    binary_report: TrainingReportData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tags with empty/None values (e.g. an unavailable git sha) are dropped."""
    monkeypatch.setattr("jabs.classifier.mlflow_logging._git_sha", lambda: None)
    tags = build_tags(binary_report)
    assert tags == {
        "behavior": "Walk",
        "classifier": "XGBoost",
        "cv_grouping_strategy": "Filename Pattern",
    }
    assert "jabs_git" not in tags


# --------------------------------------------------------------------------- #
# log_cross_validation_to_mlflow
# --------------------------------------------------------------------------- #
def test_log_cross_validation_to_mlflow(
    binary_report: TrainingReportData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run logs metrics/params/tags/artifact and returns run id + tracking URI."""
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setattr("jabs.classifier.mlflow_logging._git_sha", lambda: "abc1234")
    report_file = tmp_path / "report.md"
    report_file.write_text("# report")

    run_id, tracking_uri = log_cross_validation_to_mlflow(
        report_data=binary_report,
        report_file=report_file,
        tags={"purpose": "baseline"},
    )

    assert run_id == "run-123"
    assert tracking_uri == "file:///tmp/mlruns"
    assert fake.run_name == "Walk-cv-20260623-120000"
    assert fake.metrics["cv_accuracy_mean"] == pytest.approx(0.85)
    assert fake.params["behavior"] == "Walk"
    # user tag merges over auto tags, auto git tag preserved
    assert fake.tags["purpose"] == "baseline"
    assert fake.tags["jabs_git"] == "abc1234"
    assert fake.artifacts == [str(report_file)]


def test_log_cross_validation_skips_artifact_when_disabled(
    binary_report: TrainingReportData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With log_report_artifact=False, the report is not uploaded."""
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    report_file = tmp_path / "report.md"
    report_file.write_text("# report")

    log_cross_validation_to_mlflow(
        report_data=binary_report,
        report_file=report_file,
        log_report_artifact=False,
    )

    assert fake.artifacts == []


def test_log_cross_validation_missing_mlflow_raises(
    binary_report: TrainingReportData, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing mlflow install raises MlflowLoggingError with install guidance."""
    # Setting the module to None makes ``import mlflow`` raise ImportError.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(MlflowLoggingError, match="optional 'mlflow' dependency"):
        log_cross_validation_to_mlflow(report_data=binary_report)
