import numpy as np
from PySide6 import QtCore
from tabulate import tabulate

from jabs.types import ProjectDistanceUnit
from jabs.utils import FINAL_TRAIN_SEED


class TrainingThread(QtCore.QThread):
    """Thread used to run the training to keep the Qt main GUI thread responsive."""

    # signal so that the main GUI thread can be notified when the training is
    # complete
    training_complete = QtCore.Signal()

    # allow the thread to send a status string to the main GUI thread so that
    # we can update a status bar if we want
    current_status = QtCore.Signal(str)

    # allow the thread to send a status string to the main GUI thread so that
    # we can update a status bar if we want
    update_progress = QtCore.Signal(int)

    def __init__(self, project, classifier, behavior, k=1):
        super().__init__()
        self._project = project
        self._classifier = classifier
        self._behavior = behavior
        self._tasks_complete = 0
        self._k = k

    def run(self):
        """thread's main function

        Will get the feature set for all labeled frames, do the leave one group out train/test split,
        run the training, run the trained classifier on the test data, print some performance metrics,
        and print the most important features
        """
        self._tasks_complete = 0

        def id_processed():
            self._tasks_complete += 1
            self.update_progress.emit(self._tasks_complete)

        self.current_status.emit("Extracting Features")
        features, group_mapping = self._project.get_labeled_features(
            self._behavior, id_processed
        )

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

        # Figure out the cross validation count if all were requested
        if self._k == np.inf:
            self._k = self._classifier.get_leave_one_group_out_max(
                features["labels"], features["groups"]
            )

        if self._k > 0:
            for i, data in enumerate(data_generator):
                if i + 1 > self._k:
                    break
                self.current_status.emit(
                    f"cross validation iteration {i + 1} of {self._k}"
                )

                test_info = group_mapping[data["test_group"]]

                # train classifier, and then use it to classify our test data
                self._classifier.behavior_name = self._behavior
                self._classifier.set_project_settings(self._project)
                self._classifier.train(data)
                predictions = self._classifier.predict(data["test_data"])

                # calculate some performance metrics using the classifications
                accuracy = self._classifier.accuracy_score(
                    data["test_labels"], predictions
                )
                pr = self._classifier.precision_recall_score(
                    data["test_labels"], predictions
                )
                confusion = self._classifier.confusion_matrix(
                    data["test_labels"], predictions
                )

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
                print(f"  fbeta score {pr[2][0]:<12.8}  {pr[2][1]:<.8}")
                print(f"  support     {pr[3][0]:<12}  {pr[3][1]}")
                print("CONFUSION MATRIX:")
                print(f"{confusion}")
                print("-" * 70)
                print("Top 10 features by importance:")
                self._classifier.print_feature_importance(data["feature_names"], 10)

                # let the parent thread know that we've finished this iteration
                self._tasks_complete += 1
                self.update_progress.emit(self._tasks_complete)

            print("\n" + "=" * 70)
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
            print(f"std accuracy: {np.std(accuracies):.5}")
            print(f"mean fbeta score (behavior): {np.mean(fbeta_behavior):.5}")
            print(f"std fbeta score (behavior): {np.std(fbeta_behavior):.05}")
            print(f"mean fbeta score (not behavior): {np.mean(fbeta_notbehavior):.5}")
            print(f"std fbeta score (not behavior): {np.std(fbeta_notbehavior):.05}")
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
        self.current_status.emit("Training and saving final classifier")
        full_dataset = self._classifier.combine_data(
            features["per_frame"], features["window"]
        )
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

        self.current_status.emit("Training Complete")
        self.training_complete.emit()
