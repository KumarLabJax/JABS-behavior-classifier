import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from jabs.classifier import (
    Classifier,
    CrossValidationResult,
    TrainingReportData,
    generate_markdown_report,
    markdown_to_html,
    save_training_report,
)
from jabs.project import Project
from jabs.types import ProjectDistanceUnit
from jabs.utils import FINAL_TRAIN_SEED

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
            Emitted when training is complete, carrying the HTML-formatted training report.

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
        self._tasks_complete = 0
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
        self._tasks_complete = 0

        # Measure wall-clock time for training
        _t0_ns = time.perf_counter_ns()

        def check_termination_requested() -> None:
            if self._should_terminate:
                raise ThreadTerminatedError("Training was cancelled by the user")

        def id_processed() -> None:
            self._tasks_complete += 1
            self.update_progress.emit(self._tasks_complete)
            check_termination_requested()

        try:
            self.current_status.emit("Extracting Features")
            features, group_mapping = self._project.get_labeled_features(
                self._behavior,
                progress_callable=id_processed,
                should_terminate_callable=check_termination_requested,
            )

            # if the user requested to terminate the training while we were extracting features,
            # we should stop here
            check_termination_requested()

            self.current_status.emit("Generating train/test splits")
            data_generator = self._classifier.leave_one_group_out(
                features["per_frame"],
                features["window"],
                features["labels"],
                features["groups"],
            )

            # Collect cross-validation results for the training report
            cv_results = []
            accuracies = []
            fbeta_behavior = []
            iterations = 0

            # Figure out the cross validation count if all were requested
            if self._k == np.inf:
                self._k = self._classifier.get_leave_one_group_out_max(
                    features["labels"], features["groups"]
                )

            if self._k > 0:
                for i, data in enumerate(data_generator):
                    check_termination_requested()
                    iterations = i + 1
                    if i + 1 > self._k:
                        break
                    self.current_status.emit(f"cross validation iteration {i + 1} of {self._k}")

                    test_info = group_mapping[data["test_group"]]

                    # train classifier, and then use it to classify our test data
                    self._classifier.behavior_name = self._behavior
                    self._classifier.set_project_settings(self._project)
                    self._classifier.train(data)
                    predictions = self._classifier.predict(data["test_data"])

                    # calculate some performance metrics using the classifications
                    accuracy = self._classifier.accuracy_score(data["test_labels"], predictions)
                    pr = self._classifier.precision_recall_score(data["test_labels"], predictions)
                    confusion = self._classifier.confusion_matrix(data["test_labels"], predictions)

                    # Collect results for report
                    accuracies.append(accuracy)
                    fbeta_behavior.append(pr[2][1])

                    # Get top features for this iteration
                    top_features = self._classifier.get_feature_importance(limit=10)

                    # Store cross-validation result
                    cv_results.append(
                        CrossValidationResult(
                            iteration=i + 1,
                            test_video=test_info["video"],
                            test_identity=test_info["identity"],
                            accuracy=accuracy,
                            precision_behavior=pr[0][1],
                            precision_not_behavior=pr[0][0],
                            recall_behavior=pr[1][1],
                            recall_not_behavior=pr[1][0],
                            f1_behavior=pr[2][1],
                            support_behavior=int(pr[3][1]),
                            support_not_behavior=int(pr[3][0]),
                            confusion_matrix=confusion,
                            top_features=top_features,
                        )
                    )

                    # let the parent thread know that we've finished this iteration
                    self._tasks_complete += 1
                    self.update_progress.emit(self._tasks_complete)

            # retrain with all training data and fixed random seed before saving:
            check_termination_requested()
            self.current_status.emit("Training and saving final classifier")
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

            # Get top features from final model
            final_top_features = self._classifier.get_feature_importance(limit=20)

            self._project.save_classifier(self._classifier, self._behavior)
            self._tasks_complete += 1
            self.update_progress.emit(self._tasks_complete)

            # Calculate elapsed training time
            elapsed_ms = int((time.perf_counter_ns() - _t0_ns) // 1_000_000)

            # Get label counts
            behavior_count = int(np.sum(features["labels"] == 1))
            not_behavior_count = int(np.sum(features["labels"] == 0))

            behavior_bouts, not_behavior_bouts = self._bout_counts

            # Determine distance unit
            unit = (
                "cm"
                if self._project.feature_manager.distance_unit == ProjectDistanceUnit.CM
                else "pixel"
            )

            # Create training data object (cv_results will be empty list if k=0)
            training_data = TrainingReportData(
                behavior_name=self._behavior,
                classifier_type=self._classifier.classifier_name,
                distance_unit=unit,
                cv_results=cv_results,
                final_top_features=final_top_features,
                frames_behavior=behavior_count,
                frames_not_behavior=not_behavior_count,
                bouts_behavior=behavior_bouts,
                bouts_not_behavior=not_behavior_bouts,
                training_time_ms=elapsed_ms,
            )

            # Save markdown report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{self._behavior}_{timestamp}_training_report.md"
            report_path = self._training_log_dir / report_filename
            save_training_report(training_data, report_path)

            # Generate and emit HTML report
            markdown_content = generate_markdown_report(training_data)
            html_content = markdown_to_html(markdown_content)
            self.training_report.emit(html_content)

            # Update session tracker
            if self._k > 0:
                self._project.session_tracker.classifier_trained(
                    self._behavior,
                    self._classifier.classifier_name,
                    iterations,
                    float(np.mean(accuracies)),
                    float(np.mean(fbeta_behavior)),
                )
            else:
                # user didn't request cross validation, so we just log the training but can't include accuracy or fbeta scores
                self._project.session_tracker.classifier_trained(
                    self._behavior,
                    self._classifier.classifier_name,
                    iterations,
                )

            self.training_complete.emit(elapsed_ms)
        except Exception as e:
            # if there was an exception, we'll emit the Exception as a signal so that
            # the main GUI thread can handle it
            self.error_callback.emit(e)
