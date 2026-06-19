import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from jabs.classifier import (
    Classifier,
    MultiClassClassifier,
    generate_markdown_report,
    save_training_report,
)
from jabs.classifier.cross_validation import run_leave_one_group_out_cv
from jabs.core.constants import FINAL_TRAIN_SEED
from jabs.core.enums import ClassifierMode, ProjectDistanceUnit
from jabs.project import Project

from .exceptions import ThreadTerminatedError
from .training_strategy import (
    BinaryTrainingStrategy,
    MultiClassTrainingStrategy,
    TrainingStrategy,
)


class TrainingThread(QThread):
    """Thread used to run classifier training in the background, keeping the Qt main GUI thread responsive.

    Signals:
        training_complete: QtCore.Signal(int)
            Emitted when training is finished successfully, carrying the elapsed wall-clock time in milliseconds.
        current_status: QtCore.Signal(str)
            Emitted to update the main GUI thread with a status message (e.g., for a status bar).
        update_progress: QtCore.Signal(int)
            Emitted to inform the main GUI thread of the number of completed tasks (e.g., for a progress bar).
        error_callback: QtCore.Signal(Exception)
            Emitted if an error occurs during training, passing the exception to the main GUI thread.
        training_report: QtCore.Signal(str)
            Emitted when training is complete, carrying the Markdown-formatted training report.

    Args:
        classifier (Classifier): The classifier instance to train.
        project (Project): The project containing data and settings.
        behavior (str): The behavior label to train on.
        bout_counts (tuple[int, int]): Tuple containing counts of behavior and not-behavior bouts.
        k (int, optional): Number of cross-validation splits. Defaults to 1.
        parent (QWidget or None, optional): Optional parent widget.
    """

    training_complete = Signal(int)
    current_status = Signal(str)
    update_progress = Signal(int)
    error_callback = Signal(Exception)
    training_report = Signal(str)

    def __init__(
        self,
        classifier: Classifier | MultiClassClassifier,
        project: Project,
        behavior: str,
        bout_counts: tuple[int, int],
        k: int = 1,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self._project = project
        self._classifier = classifier
        self._behavior = behavior
        self._k = k
        self._should_terminate = False
        self._training_log_dir = project.project_paths.training_log_dir
        self._bout_counts = bout_counts

    def request_termination(self) -> None:
        """Request the thread to terminate early.

        This method sets a flag that is periodically checked by the worker thread.
        It is safe to call this method from the main Qt GUI thread. Since the flag
        is a simple boolean this is generally thread safe in CPython because
        assignment to a boolean is atomic, and therefore it does not require
        additional synchronization in this scenario.

        Could consider using QAtomicBool, but a standard bool should be fine here.
        """
        self._should_terminate = True

    def _build_strategy(self) -> TrainingStrategy:
        """Construct the per-mode training strategy for this run."""
        if self._project.settings_manager.classifier_mode == ClassifierMode.MULTICLASS:
            return MultiClassTrainingStrategy(
                classifier=self._classifier,
                project=self._project,
                behavior=self._behavior,
            )
        return BinaryTrainingStrategy(
            classifier=self._classifier,
            project=self._project,
            behavior=self._behavior,
            bout_counts=self._bout_counts,
        )

    def run(self) -> None:
        """Thread's main function.

        Get the feature set for all labeled frames, run leave-one-group-out
        cross-validation, train the final classifier on all data, save the
        classifier and training report, and emit progress/completion signals.
        """
        t0_ns = time.perf_counter_ns()
        tasks_complete = 0

        def check_termination_requested() -> None:
            if self._should_terminate:
                raise ThreadTerminatedError("Training was cancelled by the user")

        def id_processed() -> None:
            nonlocal tasks_complete
            tasks_complete += 1
            self.update_progress.emit(tasks_complete)
            check_termination_requested()

        try:
            strategy = self._build_strategy()
            settings = strategy.effective_settings()

            self.current_status.emit("Extracting Features")
            features, group_mapping = strategy.collect_features(
                progress_callable=id_processed,
                should_terminate_callable=check_termination_requested,
            )
            check_termination_requested()

            cv_results = run_leave_one_group_out_cv(
                classifier=self._classifier,
                project=self._project,
                features=features,
                group_mapping=group_mapping,
                behavior=self._behavior,
                k=self._k,
                status_callback=self.current_status.emit,
                progress_callback=id_processed,
                terminate_callback=check_termination_requested,
            )

            self.current_status.emit("Training Classifier")
            full_dataset = self._classifier.combine_data(features["per_frame"], features["window"])
            feature_names = full_dataset.columns.to_list()
            self._classifier.train(
                strategy.final_train_data(features, full_dataset, feature_names),
                random_seed=FINAL_TRAIN_SEED,
            )
            final_top_features = self._classifier.get_feature_importance(limit=20)
            strategy.save_classifier()

            elapsed_ms = int((time.perf_counter_ns() - t0_ns) // 1_000_000)
            unit = (
                "cm"
                if self._project.feature_manager.distance_unit == ProjectDistanceUnit.CM
                else "pixel"
            )
            training_data = strategy.build_report_data(
                features=features,
                cv_results=cv_results,
                final_top_features=final_top_features,
                elapsed_ms=elapsed_ms,
                timestamp=datetime.now(),
                cv_grouping_strategy=self._project.settings_manager.cv_grouping_strategy,
                cv_grouping_regex=self._project.settings_manager.cv_grouping_regex,
                distance_unit=unit,
                settings=settings,
            )

            timestamp_str = training_data.timestamp.strftime("%Y%m%d_%H%M%S")
            report_filename = f"{self._behavior}_{timestamp_str}_training_report.md"
            report_path = self._training_log_dir / report_filename
            save_training_report(training_data, report_path)

            markdown_content = generate_markdown_report(training_data)
            self.training_report.emit(markdown_content)

            if self._k > 0 and training_data.cv_results:
                accuracies = [cv.accuracy for cv in training_data.cv_results]
                self._project.session_tracker.classifier_trained(
                    self._behavior,
                    self._classifier.classifier_name,
                    len(training_data.cv_results),
                    float(np.mean(accuracies)),
                    strategy.cv_secondary_metric(training_data.cv_results),
                )
            else:
                self._project.session_tracker.classifier_trained(
                    self._behavior,
                    self._classifier.classifier_name,
                    0,
                )

            self.update_progress.emit(tasks_complete + 1)
            self.training_complete.emit(elapsed_ms)
        except Exception as e:
            self.error_callback.emit(e)
