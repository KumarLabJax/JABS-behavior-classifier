import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget
from tabulate import tabulate

from jabs.classifier import Classifier
from jabs.project import Project
from jabs.types import ProjectDistanceUnit
from jabs.utils import FINAL_TRAIN_SEED

from .exceptions import ThreadTerminatedError


class TrainingThread(QThread):
    """Thread used to run classifier training in the background, keeping the Qt main GUI thread responsive.

    Signals:
        training_complete: QtCore.Signal()
            Emitted when training is finished successfully.
        current_status: QtCore.Signal(str)
            Emitted to update the main GUI thread with a status message (e.g., for a status bar).
        update_progress: QtCore.Signal(int)
            Emitted to inform the main GUI thread of the number of completed tasks (e.g., for a progress bar).
        error_callback: QtCore.Signal(Exception)
            Emitted if an error occurs during training, passing the exception to the main GUI thread.

    Args:
        classifier (Classifier): The classifier instance to train.
        project (Project): The project containing data and settings.
        behavior (str): The behavior label to train on.
        k (int, optional): Number of cross-validation splits. Defaults to 1.
        parent (QWidget or None, optional): Optional parent widget.
    """

    training_complete = Signal()
    current_status = Signal(str)
    update_progress = Signal(int)
    error_callback = Signal(Exception)

    def __init__(
        self,
        classifier: Classifier,
        project: Project,
        behavior: str,
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

    def request_termination(self) -> None:
        """Request the thread to terminate early.

        This method sets a flag that is periodically checked by the worker thread.
        It is safe to call this method from the main Qt GUI thread. Since the flag
        is a simple boolean this is generally thread safe in CPython because
        assignment to a boolean is atomic and therefore it does not require
        additional synchronization in this scenario.

        Could consider using QAtomicBool, but a standard bool should be fine here.
        """
        self._should_terminate = True

    def run(self) -> None:
        """thread's main function

        Will get the feature set for all labeled frames, do the leave one group out train/test split,
        run the training, run the trained classifier on the test data, print some performance metrics,
        and print the most important features
        """
        self._tasks_complete = 0

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

            table_rows = []
            accuracies = []
            fbeta_behavior = []
            fbeta_notbehavior = []
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

                    table_rows.append(
                        [
                            accuracy,
                            pr[0][0],
                            pr[0][1],
                            pr[1][0],
                            pr[1][1],
                            pr[2][0],
                            pr[2][1],
                            f"{test_info['video']} [{test_info['identity']}]",
                        ]
                    )
                    accuracies.append(accuracy)
                    fbeta_behavior.append(pr[2][1])
                    fbeta_notbehavior.append(pr[2][0])

                    # print performance metrics and feature importance to console
                    print("-" * 70)
                    print(f"training iteration {i + 1}")
                    print("TEST DATA:")
                    print(f"\tVideo: {test_info['video']}")
                    print(f"\tIdentity: {test_info['identity']}")
                    print(f"ACCURACY: {accuracy * 100:.2f}%")
                    print("PRECISION RECALL:")
                    print(f"              {'not behavior':12}  behavior")
                    print(f"  precision   {pr[0][0]:<12.8}  {pr[0][1]:<.8}")
                    print(f"  recall      {pr[1][0]:<12.8}  {pr[1][1]:<.8}")
                    print(f"  F1 score    {pr[2][0]:<12.8}  {pr[2][1]:<.8}")
                    print(f"  support     {pr[3][0]:<12}  {pr[3][1]}")
                    print("CONFUSION MATRIX:")
                    print(f"{confusion}")
                    print("-" * 70)
                    print("Top 10 features by importance:")
                    self._classifier.print_feature_importance(data["feature_names"], 10)

                    # let the parent thread know that we've finished this iteration
                    self._tasks_complete += 1
                    self.update_progress.emit(self._tasks_complete)

                print("\n" + "=" * 120)
                print("SUMMARY\n")
                print(
                    tabulate(
                        table_rows,
                        showindex="always",
                        headers=[
                            "accuracy",
                            "precision\n(not behavior)",
                            "precision\n(behavior)",
                            "recall\n(not behavior)",
                            "recall\n(behavior)",
                            "f beta score\n(not behavior)",
                            "f beta score\n(behavior)",
                            "test - leave one out:\n(video [identity])",
                        ],
                    )
                )

                print(f"\nmean accuracy: {np.mean(accuracies):.5}")
                print(f"std-dev accuracy: {np.std(accuracies):.5}")
                print(f"mean F1 score (behavior): {np.mean(fbeta_behavior):.5}")
                print(f"std-dev F1 score (behavior): {np.std(fbeta_behavior):.05}")
                print(f"mean F1 score (not behavior): {np.mean(fbeta_notbehavior):.5}")
                print(f"std-dev F1 score (not behavior): {np.std(fbeta_notbehavior):.05}")
                print(f"\nClassifier: {self._classifier.classifier_name}")
                print(f"Behavior: {self._behavior}")
                # TODO: move settings print to a common project function
                # this will reduce repeated formatting across this and classify.py
                unit = (
                    "cm"
                    if self._project.feature_manager.distance_unit == ProjectDistanceUnit.CM
                    else "pixel"
                )
                print(f"Feature Distance Unit: {unit}")
                print("-" * 70)

            # retrain with all training data and fixed random seed before saving:
            check_termination_requested()
            self.current_status.emit("Training and saving final classifier")
            full_dataset = self._classifier.combine_data(features["per_frame"], features["window"])
            self._classifier.train(
                {
                    "training_data": full_dataset,
                    "training_labels": features["labels"],
                    "feature_names": full_dataset.columns.to_list(),
                },
                random_seed=FINAL_TRAIN_SEED,
            )

            self._project.save_classifier(self._classifier, self._behavior)
            self._tasks_complete += 1
            self.update_progress.emit(self._tasks_complete)

            if self._k > 0:
                self._project.session_tracker.classifier_trained(
                    self._behavior,
                    self._classifier.classifier_name,
                    iterations,
                    float(np.mean(accuracies)),
                    float(np.mean(fbeta_behavior)),
                    float(np.mean(fbeta_notbehavior)),
                )
            else:
                # user didn't request cross validation, so we just log the training but can't include accuracy or fbeta scores
                self._project.session_tracker.classifier_trained(
                    self._behavior,
                    self._classifier.classifier_name,
                    iterations,
                )

            self.current_status.emit("Training Complete")
            self.training_complete.emit()
        except Exception as e:
            # if there was an exception, we'll emit the Exception as a signal so that
            # the main GUI thread can handle it
            self.error_callback.emit(e)
