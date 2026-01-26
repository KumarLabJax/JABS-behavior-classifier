import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from jabs.classifier import (
    Classifier,
    TrainingReportData,
    generate_markdown_report,
    save_training_report,
)
from jabs.classifier.cross_validation import run_leave_one_group_out_cv
from jabs.core.constants import FINAL_TRAIN_SEED
from jabs.core.enums import ProjectDistanceUnit
from jabs.project import Project

from .exceptions import ThreadTerminatedError


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
        classifier: Classifier,
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

    def run(self) -> None:
        """thread's main function

        Will get the feature set for all labeled frames, do the leave one group out train/test split,
        run the training, run the trained classifier on the test data, collect performance metrics,
        and generate a training report (saved as markdown and emitted as HTML).
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
            self.current_status.emit("Extracting Features")
            features, group_mapping = self._project.get_labeled_features(
                self._behavior,
                progress_callable=id_processed,
                should_terminate_callable=check_termination_requested,
            )
            check_termination_requested()

            # do LOGO cross-validation
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

            # Final training on all data
            full_dataset = self._classifier.combine_data(features["per_frame"], features["window"])
            feature_names = full_dataset.columns.to_list()
            self._classifier.train(
                {
                    "training_data": full_dataset,
                    "training_labels": features["labels"],
                    "feature_names": feature_names,
                },
                random_seed=FINAL_TRAIN_SEED,
            )
            final_top_features = self._classifier.get_feature_importance(limit=20)
            self._project.save_classifier(self._classifier, self._behavior)

            # Prepare training report
            elapsed_ms = int((time.perf_counter_ns() - t0_ns) // 1_000_000)
            behavior_count = int(np.sum(features["labels"] == 1))
            not_behavior_count = int(np.sum(features["labels"] == 0))
            behavior_bouts, not_behavior_bouts = self._bout_counts
            unit = (
                "cm"
                if self._project.feature_manager.distance_unit == ProjectDistanceUnit.CM
                else "pixel"
            )
            report_timestamp = datetime.now()
            behavior_settings = self._project.settings_manager.get_behavior(self._behavior)
            training_data = TrainingReportData(
                behavior_name=self._behavior,
                classifier_type=self._classifier.classifier_name,
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
                cv_grouping_strategy=self._project.settings_manager.cv_grouping_strategy,
            )

            # Save markdown report
            timestamp_str = training_data.timestamp.strftime("%Y%m%d_%H%M%S")
            report_filename = f"{self._behavior}_{timestamp_str}_training_report.md"
            report_path = self._training_log_dir / report_filename
            save_training_report(training_data, report_path)

            # Generate and emit markdown report
            markdown_content = generate_markdown_report(training_data)
            self.training_report.emit(markdown_content)

            # Update session tracker
            if self._k > 0 and training_data.cv_results:
                accuracies = [cv.accuracy for cv in training_data.cv_results]
                fbeta_behavior = [cv.f1_behavior for cv in training_data.cv_results]
                self._project.session_tracker.classifier_trained(
                    self._behavior,
                    self._classifier.classifier_name,
                    len(training_data.cv_results),
                    float(np.mean(accuracies)),
                    float(np.mean(fbeta_behavior)),
                )
            else:
                self._project.session_tracker.classifier_trained(
                    self._behavior,
                    self._classifier.classifier_name,
                    0,
                )

            self.update_progress.emit(tasks_complete + 1)
            self.training_complete.emit(training_data.training_time_ms)
        except Exception as e:
            self.error_callback.emit(e)
