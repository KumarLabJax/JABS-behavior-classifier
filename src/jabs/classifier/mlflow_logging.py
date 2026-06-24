"""MLflow run + artifact logging for classifier cross-validation results.

Opt-in tracking for a JABS cross-validation run: one MLflow *run* per
invocation that records aggregate cross-validation metrics, a curated set of
configuration scalars as params, descriptive tags, and the generated training
report as an artifact.

Connection configuration (tracking URI, auth, TLS) is **not** hard-coded here;
it is read from standard ``MLFLOW_*`` environment variables, populated either
from a ``.env`` file (see :func:`load_env_file`) or from the ambient
environment. Each run is logged to a per-behavior experiment by default
(``jabs-<behavior>``); see :func:`resolve_experiment_name` for the override
precedence (explicit name, then ``MLFLOW_EXPERIMENT_NAME``, then the default).

``mlflow`` is an optional dependency. Install it with
``pip install 'jabs-behavior-classifier[mlflow]'`` (or, for a development
checkout, ``uv sync --extra mlflow``). :func:`log_cross_validation_to_mlflow`
raises :class:`MlflowLoggingError` with installation guidance if it is missing.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .training_report import BinaryCVResult, MultiClassCVResult

if TYPE_CHECKING:
    from .training_report import TrainingReportData

logger = logging.getLogger(__name__)


def mlflow_available() -> bool:
    """Return True if the optional ``mlflow`` package is importable.

    Uses :func:`importlib.util.find_spec` so the (heavy) ``mlflow`` package is
    not actually imported just to test for its presence. Lets callers degrade
    gracefully -- warning and skipping MLflow logging -- when the optional
    'mlflow' extra is not installed.
    """
    return importlib.util.find_spec("mlflow") is not None


class MlflowLoggingError(RuntimeError):
    """Raised when pushing cross-validation results to MLflow fails.

    The cross-validation run itself has already completed and its report has
    been saved by the time this is raised; it signals only that the optional
    MLflow push did not succeed (e.g. missing dependency, network, auth, TLS).
    """


def parse_kv_tags(items: list[str] | None) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` ``--mlflow-tag`` entries into a dict.

    Args:
        items: Raw ``KEY=VALUE`` strings (or None / empty).

    Returns:
        Mapping of tag key to value. ``None`` / empty input yields ``{}``. The
        first ``=`` splits the entry; later ``=`` characters go to the value.

    Raises:
        ValueError: If an entry has no ``=`` or an empty key.
    """
    tags: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"invalid --mlflow-tag (expected KEY=VALUE): {item!r}")
        tags[key] = value.strip()
    return tags


def load_env_file(env_file: Path | None, *, override: bool = True) -> dict[str, str]:
    """Apply the ``MLFLOW_*`` settings from a ``.env`` file to ``os.environ``.

    Only keys beginning with ``MLFLOW_`` are applied; any other keys in the file
    are ignored, so it cannot accidentally clobber unrelated environment
    variables (``PATH``, ``HTTP_PROXY``, ...). Lines may be blank, ``#``
    comments, or ``KEY=VALUE`` (an optional leading ``export`` and surrounding
    quotes on the value are stripped).

    Args:
        env_file: Path to a ``.env`` file, or None to read connection config
            from the ambient environment (a no-op returning ``{}``).
        override: If True (default), file values win for keys they define, since
            naming a file is an explicit request to use its settings.

    Returns:
        The ``MLFLOW_*`` mapping found in the file (handy for diagnostics).

    Raises:
        FileNotFoundError: If ``env_file`` is given but does not exist.
    """
    if env_file is None:
        return {}
    env_file = Path(env_file)
    if not env_file.is_file():
        raise FileNotFoundError(f"--mlflow env file not found: {env_file}")

    values: dict[str, str] = {}
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.startswith("MLFLOW_"):
            continue
        values[key] = value.strip().strip('"').strip("'")

    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def _git_sha() -> str | None:
    """Short git SHA of the jabs checkout, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    sha = result.stdout.strip()
    return sha or None


def aggregate_cv_metrics(report_data: TrainingReportData) -> dict[str, float]:
    """Aggregate per-iteration cross-validation results into scalar metrics.

    Computes the mean and population standard deviation of accuracy across CV
    iterations, plus class-specific scores (behavior precision/recall/F1 for
    binary results, macro precision/recall/F1 for multi-class results), and
    records dataset composition and timing as additional metrics.

    Args:
        report_data: Completed training report data.

    Returns:
        Mapping of MLflow-legal metric key to finite float value. Empty if
        ``report_data`` has no cross-validation iterations.
    """
    cv_results = report_data.cv_results
    if not cv_results:
        return {}

    metrics: dict[str, float] = {}
    accuracy = np.array([r.accuracy for r in cv_results], dtype=float)
    metrics["cv_accuracy_mean"] = float(np.mean(accuracy))
    metrics["cv_accuracy_std"] = float(np.std(accuracy))
    metrics["cv_iterations"] = float(len(cv_results))

    if all(isinstance(r, BinaryCVResult) for r in cv_results):
        per_class_attrs = ("precision_behavior", "recall_behavior", "f1_behavior")
    elif all(isinstance(r, MultiClassCVResult) for r in cv_results):
        per_class_attrs = ("precision_macro", "recall_macro", "f1_macro")
    else:
        per_class_attrs = ()

    for attr in per_class_attrs:
        vals = np.array([getattr(r, attr) for r in cv_results], dtype=float)
        metrics[f"cv_{attr}_mean"] = float(np.mean(vals))
        metrics[f"cv_{attr}_std"] = float(np.std(vals))

    metrics["frames_behavior"] = float(report_data.frames_behavior)
    metrics["frames_not_behavior"] = float(report_data.frames_not_behavior)
    metrics["bouts_behavior"] = float(report_data.bouts_behavior)
    metrics["bouts_not_behavior"] = float(report_data.bouts_not_behavior)
    metrics["training_time_ms"] = float(report_data.training_time_ms)

    return {key: value for key, value in metrics.items() if math.isfinite(value)}


def build_params(report_data: TrainingReportData) -> dict[str, str]:
    """Build the curated, filterable MLflow params for a CV run.

    These are the columns you sort/filter the leaderboard by. The full results
    ride the training-report artifact.

    Args:
        report_data: Completed training report data.

    Returns:
        Mapping of param key to stringified value.
    """
    params: dict[str, object] = {
        "behavior": report_data.behavior_name,
        "classifier": report_data.classifier_type,
        "window_size": report_data.window_size,
        "balance_labels": report_data.balance_training_labels,
        "symmetric_behavior": report_data.symmetric_behavior,
        "distance_unit": report_data.distance_unit,
        "cv_grouping_strategy": report_data.cv_grouping_strategy.value,
    }
    if report_data.cv_grouping_regex:
        params["cv_grouping_regex"] = report_data.cv_grouping_regex
    return {key: str(value) for key, value in params.items()}


def build_tags(report_data: TrainingReportData) -> dict[str, str]:
    """Build the auto-derived MLflow run tags for a CV run.

    Args:
        report_data: Completed training report data.

    Returns:
        Mapping of tag key to value, omitting any whose value is empty/None.
    """
    tags = {
        "behavior": report_data.behavior_name,
        "classifier": report_data.classifier_type,
        "cv_grouping_strategy": report_data.cv_grouping_strategy.value,
        "jabs_git": _git_sha(),
    }
    return {key: value for key, value in tags.items() if value}


def resolve_experiment_name(report_data: TrainingReportData, experiment_name: str | None) -> str:
    """Resolve the MLflow experiment name for a cross-validation run.

    Each behavior is logged to its own experiment by default so that runs of the
    same behavior are compared together (and not mixed with other behaviors, whose
    metrics are not comparable). Precedence, highest first:

    1. ``experiment_name`` -- an explicit override (e.g. from ``--mlflow-experiment``).
    2. The ``MLFLOW_EXPERIMENT_NAME`` environment variable, if set.
    3. The default ``jabs-<behavior>``.

    Args:
        report_data: Completed training report data (supplies the behavior name).
        experiment_name: Explicit override, or None to fall back to the env var/default.

    Returns:
        The experiment name to use.
    """
    return (
        experiment_name
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or f"jabs-{report_data.behavior_name}"
    )


def log_cross_validation_to_mlflow(
    *,
    report_data: TrainingReportData,
    report_file: Path | None = None,
    env_file: Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    log_report_artifact: bool = True,
) -> tuple[str, str]:
    """Create one MLflow run for a cross-validation run and return its ids.

    Logs aggregate CV metrics, curated params, auto-derived plus caller tags,
    and (optionally) the training report as an artifact. Connection config comes
    from the environment; ``env_file``, if given, is loaded into it first.

    Args:
        report_data: Completed training report data.
        report_file: Path to the saved training report to upload as an artifact.
            Ignored if None or missing on disk, or if ``log_report_artifact`` is
            False.
        env_file: Optional ``.env`` file with ``MLFLOW_*`` connection settings.
            If None, connection config comes from the ambient environment.
        experiment_name: Explicit MLflow experiment name. If None, the experiment is
            resolved by :func:`resolve_experiment_name` (``MLFLOW_EXPERIMENT_NAME`` env
            var, else the default ``jabs-<behavior>``). The experiment is created if it
            does not exist.
        run_name: MLflow run name. Defaults to ``<behavior>-cv-<timestamp>``, where the
            timestamp is the report's completion time (so it matches the saved report).
        tags: Caller-supplied run tags; merged over the auto-derived tags (so a
            user tag with the same key wins).
        log_report_artifact: Whether to upload the training report artifact.

    Returns:
        A ``(run_id, tracking_uri)`` tuple for the created MLflow run.

    Raises:
        MlflowLoggingError: If the ``mlflow`` package is not installed.
    """
    try:
        import mlflow
    except ImportError as e:
        raise MlflowLoggingError(
            "MLflow logging requires the optional 'mlflow' dependency. Install it with "
            "`pip install 'jabs-behavior-classifier[mlflow]'` "
            "(or, for a development checkout, `uv sync --extra mlflow`)."
        ) from e

    load_env_file(env_file)

    resolved_experiment = resolve_experiment_name(report_data, experiment_name)
    mlflow.set_experiment(resolved_experiment)

    if run_name is None:
        run_name = f"{report_data.behavior_name}-cv-{report_data.timestamp:%Y%m%d-%H%M%S}"

    logger.info(
        "Logging cross-validation results to MLflow experiment %r run %r",
        resolved_experiment,
        run_name,
    )
    with mlflow.start_run(run_name=run_name) as run:
        metrics = aggregate_cv_metrics(report_data)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        params = build_params(report_data)
        if params:
            mlflow.log_params(params)

        merged_tags = build_tags(report_data)
        merged_tags.update(tags or {})
        if merged_tags:
            mlflow.set_tags(merged_tags)

        if log_report_artifact and report_file is not None and Path(report_file).is_file():
            mlflow.log_artifact(str(report_file))

        run_id = run.info.run_id

    tracking_uri = mlflow.get_tracking_uri()
    logger.info("Logged cross-validation results to MLflow run %s (%s)", run_id, tracking_uri)
    return run_id, tracking_uri
