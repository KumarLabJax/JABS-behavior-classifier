import time
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from jabs.classifier import (
    Classifier,
    TrainingReportData,
    run_leave_one_group_out_cv,
    save_training_report,
)
from jabs.constants import FINAL_TRAIN_SEED
from jabs.enums import ClassifierType, CrossValidationGroupingStrategy, ProjectDistanceUnit
from jabs.project import Project

N_JOBS = 4


def run_cross_validation(
    project_dir: Path,
    behavior: str,
    classifier_type: ClassifierType,
    grouping_strategy: CrossValidationGroupingStrategy | None,
    k: int,
    report_file: Path | None = None,
) -> None:
    """Run cross-validation for a JABS project from the command line.

    Prints results to the console and saves a training report markdown file.

    Args:
        project_dir (Path): Path to the JABS project directory.
        behavior (str): Behavior label to perform cross-validation on.
        classifier_type (ClassifierType): Classifier type to use.
        grouping_strategy (CrossValidationGroupingStrategy): Grouping strategy for cross-validation.
         If None, uses project settings.
        k (int): Number of cross-validation splits. Use 0 for max splits.
        report_file (Path | None): Path to save the training report markdown file.
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

    # validate the behavior
    if behavior not in project.settings_manager.behavior_names:
        raise ValueError(f"The specified behavior '{behavior}' is not found in the project.")

    classifier = Classifier(classifier=classifier_type, n_jobs=N_JOBS)

    console = Console()
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

    def status_callback(msg: str):
        nonlocal status_message
        status_message = msg
        console.status(msg)

    def progress_callback():
        if progress.tasks:
            progress.advance(task_id)

    t0_ns = time.perf_counter_ns()

    with console.status("Extracting features for labeled frames...", spinner="dots"):
        features, group_mapping = project.get_labeled_features(
            behavior,
            grouping_strategy=grouping_strategy,
        )

    with progress:
        if k == 0:
            k = classifier.get_leave_one_group_out_max(features["labels"], features["groups"])

        task_id = progress.add_task(f"Cross-validation ({behavior})", total=k)
        cv_results = run_leave_one_group_out_cv(
            classifier=classifier,
            project=project,
            features=features,
            group_mapping=group_mapping,
            behavior=behavior,
            k=k,
            status_callback=status_callback,
            progress_callback=progress_callback,
        )
    console.print(f"Cross-validation complete. {len(cv_results)} iterations performed.")

    # Print Rich table of results
    if cv_results:
        table = Table(title="Cross-Validation Results")
        table.add_column("Iter", justify="center")
        table.add_column("Accuracy", justify="right")
        table.add_column("Precision\n(Behavior)", justify="right")
        table.add_column("Precision\n(Not Behavior)", justify="right")
        table.add_column("Recall\n(Behavior)", justify="right")
        table.add_column("Recall\n(Not Behavior)", justify="right")
        table.add_column("F1 Score", justify="right")
        table.add_column("Test Group", justify="left")
        for cv in cv_results:
            table.add_row(
                str(cv.iteration),
                f"{cv.accuracy:.3f}",
                f"{cv.precision_behavior:.3f}",
                f"{cv.precision_not_behavior:.3f}",
                f"{cv.recall_behavior:.3f}",
                f"{cv.recall_not_behavior:.3f}",
                f"{cv.f1_behavior:.3f}",
                str(cv.test_label),
            )
        console.print(table)

    # train final model on all data
    with console.status(
        "Training final model on all labeled data for feature importance...", spinner="dots"
    ):
        features, _ = project.get_labeled_features(behavior)
        full_dataset = classifier.combine_data(features["per_frame"], features["window"])
        feature_names = full_dataset.columns.to_list()
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

    # get bout counts
    behavior_bouts = 0
    not_behavior_bouts = 0
    for _video, video_counts in project.counts(behavior).items():
        for _identity, counts in video_counts.items():
            behavior_bouts += counts["unfragmented_bout_counts"][0]
            not_behavior_bouts += counts["unfragmented_bout_counts"][1]

    # get labeled frame counts
    behavior_count = int(np.sum(features["labels"] == 1))
    not_behavior_count = int(np.sum(features["labels"] == 0))

    unit = "cm" if project.feature_manager.distance_unit == ProjectDistanceUnit.CM else "pixel"
    report_timestamp = datetime.now()
    behavior_settings = project.settings_manager.get_behavior(behavior)
    training_data = TrainingReportData(
        behavior_name=behavior,
        classifier_type=classifier.classifier_name,
        balance_training_labels=behavior_settings.get("balance_labels", False),
        symmetric_behavior=behavior_settings.get("symmetric_behavior", False),
        distance_unit=unit,
        cv_results=cv_results,
        final_top_features=final_top_features,
        frames_behavior=behavior_count,
        frames_not_behavior=not_behavior_count,
        bouts_behavior=behavior_bouts,
        bouts_not_behavior=not_behavior_bouts,
        training_time_ms=elapsed_ms,
        timestamp=report_timestamp,
        window_size=behavior_settings["window_size"],
        cv_grouping_strategy=project.settings_manager.cv_grouping_strategy,
    )

    # Save markdown report
    if report_file is None:
        timestamp_str = training_data.timestamp.strftime("%Y%m%d_%H%M%S")
        report_file = Path(f"{behavior}_{timestamp_str}_training_report.md")

    save_training_report(training_data, report_file)
    console.print(f"\nTraining report saved to: {report_file}", style="bold green")
