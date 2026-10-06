import time
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from jabs.classifier import (
    NO_VALID_SPLITS_WARNING,
    BinaryCVResult,
    Classifier,
    CrossValidationResult,
    MlflowLoggingError,
    MultiClassClassifier,
    MultiClassCVResult,
    TrainingReportData,
    classifier_utils,
    enabled_stage_configs,
    log_cross_validation_to_mlflow,
    postprocessed_results,
    run_leave_one_group_out_cv,
    save_training_report,
)
from jabs.classifier.cross_validation import CVFeatures
from jabs.core.constants import FINAL_TRAIN_SEED
from jabs.core.enums import (
    ClassifierMode,
    ClassifierType,
    CrossValidationGroupingStrategy,
    ProjectDistanceUnit,
)
from jabs.project import Project

N_JOBS = 4

# Multi-class cross-validation covers every behavior at once, so reports and MLflow
# runs are named for the mode rather than for a single behavior.
MULTICLASS_REPORT_NAME = "multiclass"


def _included_row_mask(features: CVFeatures) -> npt.NDArray[np.bool_] | None:
    """Select the feature rows whose group is not excluded from training.

    Videos excluded from training still appear in ``features`` so they can serve as
    held-out cross-validation groups, but the final model must not train on them.

    Args:
        features: Feature payload from ``Project.get_multiclass_labeled_features``.

    Returns:
        A boolean mask aligned to the feature rows, or None when no group is excluded.
    """
    excluded = features.get("excluded_groups")
    if not excluded:
        return None
    return ~np.isin(features["groups"], list(excluded))


def _max_multiclass_splits(classifier: MultiClassClassifier, features: CVFeatures) -> int:
    """Count the valid leave-one-group-out splits for multi-class features.

    Args:
        classifier: Multi-class classifier the splits are validated against.
        features: Feature payload from ``Project.get_multiclass_labeled_features``.

    Returns:
        Number of groups that can serve as a valid test split.
    """
    labels_by_behavior = features["labels_by_behavior"]
    if not labels_by_behavior:
        return 0
    labels, _ = classifier_utils.merge_labels(labels_by_behavior, classifier.behavior_names)
    return classifier.get_leave_one_group_out_max(
        labels, features["groups"], features.get("excluded_groups")
    )


def _train_final_multiclass(
    classifier: MultiClassClassifier,
    features: CVFeatures,
    settings: dict,
) -> list[tuple[str, float]]:
    """Train the multi-class classifier on all included labeled data.

    Args:
        classifier: Multi-class classifier to train in place.
        features: Feature payload from ``Project.get_multiclass_labeled_features``.
        settings: Effective training settings (window size, balancing, ...).

    Returns:
        The classifier's top 10 ``(feature name, importance)`` pairs.
    """
    mask = _included_row_mask(features)
    per_frame = features["per_frame"]
    window = features["window"]
    labels_by_behavior = features["labels_by_behavior"]
    if mask is not None:
        per_frame = per_frame[mask].reset_index(drop=True)
        window = window[mask].reset_index(drop=True)
        labels_by_behavior = {name: arr[mask] for name, arr in labels_by_behavior.items()}

    # cross-validation folds pass the settings in their payload; a run with no valid
    # splits reaches the final fit without having used them
    classifier.set_dict_settings(settings)
    feature_names = classifier.combine_data(per_frame, window).columns.to_list()
    classifier.train(
        {
            "per_frame": per_frame,
            "window": window,
            "labels_by_behavior": labels_by_behavior,
            "settings": settings,
            "feature_names": feature_names,
        },
        random_seed=FINAL_TRAIN_SEED,
    )
    return classifier.get_feature_importance(limit=10)


def _multiclass_class_counts(
    project: Project,
    classifier: MultiClassClassifier,
    features: CVFeatures,
) -> tuple[dict[str, int], dict[str, int]]:
    """Count labeled frames and bouts per class over the videos trained on.

    Args:
        project: Project the labels come from.
        classifier: Multi-class classifier supplying the ordered class names.
        features: Feature payload from ``Project.get_multiclass_labeled_features``.

    Returns:
        ``(frame counts, bout counts)``, each keyed by class name (including the
        reserved None class).
    """
    class_names = classifier.get_class_names()
    mask = _included_row_mask(features)
    labels_by_behavior = features["labels_by_behavior"]
    if mask is not None:
        labels_by_behavior = {name: arr[mask] for name, arr in labels_by_behavior.items()}
    merged_labels, _ = classifier_utils.merge_labels(labels_by_behavior, classifier.behavior_names)
    frame_counts = {
        name: int(np.sum(merged_labels == class_idx)) for class_idx, name in enumerate(class_names)
    }

    settings_manager = project.settings_manager
    bout_counts: dict[str, int] = {}
    for class_name in class_names:
        bouts = 0
        for video, video_counts in project.counts(class_name).items():
            if settings_manager.is_video_excluded(video):
                continue
            for identity_counts in video_counts.values():
                bouts += identity_counts["unfragmented_bout_counts"][0]
        bout_counts[class_name] = bouts
    return frame_counts, bout_counts


def _print_consistency_warnings(console: Console, cv_results: list[CrossValidationResult]) -> None:
    """Print any iteration whose two prediction passes disagreed.

    The results table puts raw and postprocessed metrics side by side, so it
    has to say when that comparison is not meaningful.

    Args:
        console: Rich console to print to.
        cv_results: Cross-validation iteration results.
    """
    for cv in postprocessed_results(cv_results):
        if cv.postprocessed.consistency_warning:
            console.print(
                f"[yellow]Warning (iteration {cv.iteration}):[/yellow] "
                f"{cv.postprocessed.consistency_warning}"
            )


def _print_multiclass_results(console: Console, cv_results: list[CrossValidationResult]) -> None:
    """Print the per-iteration table for multi-class cross-validation.

    Columns match the multi-class table in the training report.

    Args:
        console: Rich console to print to.
        cv_results: Cross-validation iteration results. Does nothing when empty.
    """
    if not cv_results:
        return
    table = Table(title="Cross-Validation Results")
    table.add_column("Iter", justify="center")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision\n(Macro)", justify="right")
    table.add_column("Recall\n(Macro)", justify="right")
    table.add_column("F1\n(Macro)", justify="right")
    table.add_column("F1\n(Micro)", justify="right")
    table.add_column("Test Group", justify="left")
    for cv in cv_results:
        if not isinstance(cv, MultiClassCVResult):
            continue
        table.add_row(
            str(cv.iteration),
            f"{cv.accuracy:.3f}",
            f"{cv.precision_macro:.3f}",
            f"{cv.recall_macro:.3f}",
            f"{cv.f1_macro:.3f}",
            f"{cv.f1_micro:.3f}",
            str(cv.test_label),
        )
    console.print(table)


def run_cross_validation(
    project_dir: Path,
    behavior: str | None,
    classifier_type: ClassifierType,
    grouping_strategy: CrossValidationGroupingStrategy | None,
    k: int,
    report_file: Path | None = None,
    grouping_regex: str | None = None,
    evaluate_postprocessing: bool | None = None,
    mlflow_enabled: bool = False,
    mlflow_env_file: Path | None = None,
    mlflow_experiment: str | None = None,
    mlflow_tags: dict[str, str] | None = None,
    mlflow_log_report: bool = True,
    mlflow_log_annotations: bool = True,
) -> None:
    """Run cross-validation for a JABS project from the command line.

    Prints results to the console and saves a training report markdown file. A binary
    project is cross-validated for one behavior; a multi-class project is cross-validated
    for all of its behaviors together, as the GUI does.

    Args:
        project_dir (Path): Path to the JABS project directory.
        behavior (str | None): Behavior label to perform cross-validation on. Required for
          binary projects. Ignored (with a warning) for multi-class projects.
        classifier_type (ClassifierType): Classifier type to use.
        grouping_strategy (CrossValidationGroupingStrategy): Grouping strategy for cross-validation.
          If None, uses project settings.
        k (int): Number of cross-validation splits. Use 0 for max splits.
        report_file (Path | None): Path to save the training report file.
          Format will be determined by the extension (.md for markdown or .json for JSON).
        grouping_regex (str | None): Regular expression used to extract a grouping key
          from each video filename. Only used when ``grouping_strategy`` is
          ``FILENAME_PATTERN``. If None, uses the pattern saved in project settings.
        evaluate_postprocessing (bool | None): If True, also report metrics with the
          behavior's prediction postprocessing pipeline applied. This re-predicts each
          held-out group's full tracks, so it costs roughly one classification pass over
          the labeled identities. If None, uses the behavior's saved project setting.
          Binary projects only; multi-class projects skip it (with a warning if True).
        mlflow_enabled (bool): If True, push the cross-validation results to MLflow
          after the report is saved. Callers should only enable this when the optional
          'mlflow' dependency is installed (the CLI checks this and fails fast with an
          error before running when --mlflow is requested without the extra installed).
        mlflow_env_file (Path | None): Optional ``.env`` file with ``MLFLOW_*`` connection
          settings. If None, connection config comes from the ambient environment.
        mlflow_experiment (str | None): Explicit MLflow experiment name. If None, defaults
          to the ``MLFLOW_EXPERIMENT_NAME`` env var, else ``jabs-<behavior>``.
        mlflow_tags (dict[str, str] | None): Optional free-form MLflow run tags, merged
          over the auto-derived tags.
        mlflow_log_report (bool): Whether to upload the training report as an MLflow
          artifact. Only used when ``mlflow_enabled`` is True.
        mlflow_log_annotations (bool): Whether to upload a zip of the project's
          ``jabs/annotations`` directory as an MLflow artifact, capturing the labels
          the run was computed from. Only used when ``mlflow_enabled`` is True.

    Raises:
        MlflowLoggingError: If MLflow logging is requested but fails. The
          cross-validation results and the saved report are unaffected.
    """
    if k < 0:
        raise ValueError("The number of cross-validation splits 'k' must be non-negative.")

    # validate the jabs project directory
    if not project_dir.is_dir():
        raise ValueError(f"The specified path is not a directory: {project_dir}")

    if not Project.is_valid_project_directory(project_dir):
        raise ValueError(
            f"The specified directory is not a valid JABS project directory: {project_dir}"
        )

    # load the project
    project = Project(project_dir, enable_session_tracker=False)

    console = Console()
    is_multiclass = project.settings_manager.classifier_mode == ClassifierMode.MULTICLASS
    classifier: Classifier | MultiClassClassifier
    settings: dict

    if is_multiclass:
        behavior_names = list(project.settings_manager.behavior_names)
        if not behavior_names:
            raise ValueError(
                "The project has no behaviors defined, so there is nothing to classify."
            )
        if behavior is not None:
            console.print(
                "[yellow]Warning:[/yellow] --behavior is ignored for multi-class projects; "
                "all behaviors are cross-validated together."
            )
        if evaluate_postprocessing:
            console.print(
                "[yellow]Warning:[/yellow] postprocessing evaluation is not supported for "
                "multi-class projects and will be skipped."
            )
        # prediction postprocessing is binary-only
        evaluate_postprocessing = False
        report_name = MULTICLASS_REPORT_NAME
        classifier = MultiClassClassifier(
            behavior_names, classifier_type=classifier_type, n_jobs=N_JOBS
        )
        settings = classifier.project_settings or project.get_project_defaults()
    else:
        if behavior is None:
            raise ValueError("--behavior is required for a binary classifier project.")
        # validate the behavior
        if behavior not in project.settings_manager.behavior_names:
            raise ValueError(f"The specified behavior '{behavior}' is not found in the project.")
        report_name = behavior
        classifier = Classifier(classifier=classifier_type, n_jobs=N_JOBS)
        settings = project.settings_manager.get_behavior(behavior)

        # None means "use the behavior's saved setting", matching how the grouping
        # strategy and pattern overrides work.
        if evaluate_postprocessing is None:
            evaluate_postprocessing = project.settings_manager.evaluate_postprocessing_in_cv(
                behavior
            )

    status_message = "Starting cross-validation..."
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    task_id = None
    cv_warning: str | None = None

    def status_callback(msg: str):
        nonlocal status_message
        status_message = msg
        console.status(msg)

    def warning_callback(msg: str):
        nonlocal cv_warning
        cv_warning = msg

    def progress_callback():
        if progress.tasks:
            progress.advance(task_id)

    t0_ns = time.perf_counter_ns()

    with console.status("Extracting features for labeled frames...", spinner="dots"):
        if is_multiclass:
            features, group_mapping = project.get_multiclass_labeled_features(
                grouping_strategy=grouping_strategy,
                grouping_regex=grouping_regex,
                behavior_settings=settings,
            )
        else:
            features, group_mapping = project.get_labeled_features(
                behavior,
                grouping_strategy=grouping_strategy,
                grouping_regex=grouping_regex,
            )

    with progress:
        if k == 0:
            # k=0 means "as many splits as the data supports" here, so a maximum of
            # zero is not a request for no cross-validation - it is a failure to find
            # any, which run_leave_one_group_out_cv would not warn about.
            if is_multiclass:
                k = _max_multiclass_splits(classifier, features)
            else:
                k = classifier.get_leave_one_group_out_max(features["labels"], features["groups"])
            if k == 0:
                warning_callback(NO_VALID_SPLITS_WARNING)

        task_id = progress.add_task(f"Cross-validation ({report_name})", total=k)
        cv_results = run_leave_one_group_out_cv(
            classifier=classifier,
            project=project,
            features=features,
            group_mapping=group_mapping,
            behavior=report_name,
            k=k,
            status_callback=status_callback,
            progress_callback=progress_callback,
            warning_callback=warning_callback,
            evaluate_postprocessing=evaluate_postprocessing,
        )
    console.print(f"Cross-validation complete. {len(cv_results)} iterations performed.")
    if cv_warning:
        console.print(f"[yellow]Warning:[/yellow] {cv_warning}")

    # Print Rich table of results
    if is_multiclass:
        _print_multiclass_results(console, cv_results)
    elif cv_results:
        # same predicate the markdown and JSON reports use, so the three
        # surfaces cannot disagree about which iterations have these metrics
        show_postprocessed = bool(postprocessed_results(cv_results))
        table = Table(title="Cross-Validation Results")
        table.add_column("Iter", justify="center")
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision\n(Behavior)", justify="right")
        table.add_column("Precision\n(Not Behavior)", justify="right")
        table.add_column("Recall\n(Behavior)", justify="right")
        table.add_column("Recall\n(Not Behavior)", justify="right")
        table.add_column("F1 Score", justify="right")
        if show_postprocessed:
            # placed next to the raw values they should be compared against
            table.add_column("Accuracy\n(Postproc.)", justify="right")
            table.add_column("F1 Score\n(Postproc.)", justify="right")
        table.add_column("Test Group", justify="left")
        for cv in cv_results:
            row = [
                str(cv.iteration),
                f"{cv.accuracy:.3f}",
                f"{cv.precision_behavior:.3f}",
                f"{cv.precision_not_behavior:.3f}",
                f"{cv.recall_behavior:.3f}",
                f"{cv.recall_not_behavior:.3f}",
                f"{cv.f1_behavior:.3f}",
            ]
            if show_postprocessed:
                postprocessed = cv.postprocessed if isinstance(cv, BinaryCVResult) else None
                row.append(f"{postprocessed.accuracy:.3f}" if postprocessed else "-")
                row.append(f"{postprocessed.f1_behavior:.3f}" if postprocessed else "-")
            row.append(str(cv.test_label))
            table.add_row(*row)
        console.print(table)
        _print_consistency_warnings(console, cv_results)

        if not show_postprocessed and evaluate_postprocessing:
            console.print(
                "[yellow]Postprocessing evaluation was requested but produced no "
                "metrics (no enabled stages, or no scorable held-out frames).[/yellow]"
            )

    # train final model on all data
    with console.status(
        "Training final model on all labeled data for feature importance...", spinner="dots"
    ):
        if is_multiclass:
            # the features collected for cross-validation already cover every behavior
            final_top_features = _train_final_multiclass(classifier, features, settings)
        else:
            features, _ = project.get_labeled_features(behavior)
            full_dataset = classifier.combine_data(features["per_frame"], features["window"])
            feature_names = full_dataset.columns.to_list()
            # cross-validation folds set these as a side effect, but a run with no
            # valid splits reaches the final fit without them
            classifier.behavior_name = behavior
            classifier.set_project_settings(project, behavior)
            classifier.train(
                {
                    "training_data": full_dataset,
                    "training_labels": features["labels"],
                    "feature_names": feature_names,
                },
                random_seed=FINAL_TRAIN_SEED,
            )
            final_top_features = classifier.get_feature_importance(limit=10)

    # output final top features
    console.print("\nTop 10 Features from Final Model Trained on All Data:")
    feature_table = Table(title="Final Model Feature Importance")
    feature_table.add_column("Rank", justify="right")
    feature_table.add_column("Feature Name", justify="left")
    feature_table.add_column("Importance", justify="right")
    for rank, (feature, importance) in enumerate(final_top_features, start=1):
        feature_table.add_row(str(rank), feature, f"{importance:.2f}")
    console.print(feature_table)

    # Prepare training report
    elapsed_ms = int((time.perf_counter_ns() - t0_ns) // 1_000_000)

    behavior_bouts = 0
    not_behavior_bouts = 0
    behavior_count = 0
    not_behavior_count = 0
    class_frame_counts: dict[str, int] | None = None
    class_bout_counts: dict[str, int] | None = None
    if is_multiclass:
        class_frame_counts, class_bout_counts = _multiclass_class_counts(
            project, classifier, features
        )
    else:
        # get bout counts
        for _video, video_counts in project.counts(behavior).items():
            for _identity, counts in video_counts.items():
                behavior_bouts += counts["unfragmented_bout_counts"][0]
                not_behavior_bouts += counts["unfragmented_bout_counts"][1]

        # get labeled frame counts
        behavior_count = int(np.sum(features["labels"] == 1))
        not_behavior_count = int(np.sum(features["labels"] == 0))

    unit = "cm" if project.feature_manager.distance_unit == ProjectDistanceUnit.CM else "pixel"
    report_timestamp = datetime.now()

    # resolve the grouping strategy/regex actually used so the report reflects any
    # command-line overrides rather than the project's saved settings.
    effective_grouping_strategy = (
        grouping_strategy
        if grouping_strategy is not None
        else project.settings_manager.cv_grouping_strategy
    )
    effective_grouping_regex = (
        grouping_regex
        if grouping_regex is not None
        else project.settings_manager.cv_grouping_regex
    )
    training_data = TrainingReportData(
        behavior_name=report_name,
        classifier_type=classifier.classifier_name,
        balance_training_labels=settings.get("balance_labels", False),
        symmetric_behavior=settings.get("symmetric_behavior", False),
        distance_unit=unit,
        cv_results=cv_results,
        cv_warning=cv_warning,
        final_top_features=final_top_features,
        frames_behavior=behavior_count,
        frames_not_behavior=not_behavior_count,
        bouts_behavior=behavior_bouts,
        bouts_not_behavior=not_behavior_bouts,
        class_frame_counts=class_frame_counts,
        class_bout_counts=class_bout_counts,
        training_time_ms=elapsed_ms,
        timestamp=report_timestamp,
        window_size=settings["window_size"],
        cv_grouping_strategy=effective_grouping_strategy,
        cv_grouping_regex=(
            effective_grouping_regex
            if effective_grouping_strategy == CrossValidationGroupingStrategy.FILENAME_PATTERN
            else None
        ),
        postprocessing_stages=(
            enabled_stage_configs(project.settings_manager.postprocessing_config(behavior))
            if evaluate_postprocessing
            else None
        ),
    )

    # Save markdown report
    if report_file is None:
        # no filename specified, generate default
        timestamp_str = training_data.timestamp.strftime("%Y%m%d_%H%M%S")
        report_file = Path(f"{report_name}_{timestamp_str}_training_report.md")

    save_training_report(training_data, report_file)
    console.print(f"\nTraining report saved to: {report_file}", style="bold green")

    # Push results to MLflow last, so a logging failure (missing dependency,
    # network, auth, TLS) never costs the cross-validation results -- they are
    # already on screen and the report is already saved.
    if mlflow_enabled:
        try:
            run_id, tracking_uri = log_cross_validation_to_mlflow(
                report_data=training_data,
                report_file=report_file,
                annotations_dir=project.annotation_dir,
                env_file=mlflow_env_file,
                experiment_name=mlflow_experiment,
                tags=mlflow_tags,
                log_report_artifact=mlflow_log_report,
                log_annotations_artifact=mlflow_log_annotations,
            )
        except Exception as e:
            console.print(f"\nWarning: MLflow logging failed: {e}", style="bold yellow")
            console.print(
                "  (cross-validation results above and the saved report are unaffected)",
                style="yellow",
            )
            # Preserve an MlflowLoggingError raised by the logger (e.g. missing
            # dependency); only wrap genuinely unexpected exceptions.
            if isinstance(e, MlflowLoggingError):
                raise
            raise MlflowLoggingError(str(e)) from e
        console.print(
            f"\nLogged cross-validation results to MLflow run {run_id} ({tracking_uri})",
            style="bold green",
        )
