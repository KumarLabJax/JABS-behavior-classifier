import itertools

import numpy as np
from PySide2 import QtCore
from tabulate import tabulate

from src.project import ProjectDistanceUnit
from src.utils import FINAL_TRAIN_SEED


class TrainingThread(QtCore.QThread):
    """
    Thread used to run the training to keep the Qt main GUI thread responsive.
    """

    # signal so that the main GUI thread can be notified when the training is
    # complete
    training_complete = QtCore.Signal()

    # allow the thread to send a status string to the main GUI thread so that
    # we can update a status bar if we want
    current_status = QtCore.Signal(str)

    # allow the thread to send a status string to the main GUI thread so that
    # we can update a status bar if we want
    update_progress = QtCore.Signal(int)

    def __init__(self, project, classifier, behavior, window_size, uses_social, uses_balance,
                 k=1):
        super().__init__()
        self._project = project
        self._classifier = classifier
        self._behavior = behavior
        self._tasks_complete = 0
        self._window_size = window_size
        self._uses_social = uses_social
        self._uses_balance = uses_balance
        self._k = k

    def run(self):
        """
        thread's main function. Will get the feature set for all labeled frames,
        do the leave one group out train/test split, run the training, run the
        trained classifier on the test data, print some performance metrics,
        and print the most important features
        """

        self._tasks_complete = 0

        def id_processed():
            self._tasks_complete += 1
            self.update_progress.emit(self._tasks_complete)

        self.current_status.emit("Extracting Features")
        features, group_mapping = self._project.get_labeled_features(
            self._behavior,
            self._window_size,
            self._uses_social,
            id_processed
        )

        self.current_status.emit("Generating train/test splits")
        data_generator = self._classifier.leave_one_group_out(
            features['per_frame'],
            features['window'],
            features['labels'],
            features['groups']
        )

        table_rows = []
        accuracies = []
        fbeta_behavior = []
        fbeta_notbehavior = []

        if self._k > 0:

            for i, data in enumerate(itertools.islice(data_generator, self._k)):
                self.current_status.emit(f"cross validation iteration {i}")

                test_info = group_mapping[data['test_group']]

                # train classifier, and then use it to classify our test data
                self._classifier.train(data, features['column_names'], self._behavior, self._window_size,
                                       self._uses_social, self._uses_balance,
                                       self._project.extended_features,
                                       self._project.distance_unit)
                predictions = self._classifier.predict(data['test_data'])

                # calculate some performance metrics using the classifications
                accuracy = self._classifier.accuracy_score(
                    data['test_labels'], predictions)
                pr = self._classifier.precision_recall_score(
                    data['test_labels'], predictions)
                confusion = self._classifier.confusion_matrix(
                    data['test_labels'], predictions)

                table_rows.append([
                    accuracy, pr[0][0], pr[0][1], pr[1][0], pr[1][1], pr[2][0],
                    pr[2][1], f"{test_info['video']} [{test_info['identity']}]"
                ])
                accuracies.append(accuracy)
                fbeta_behavior.append(pr[2][1])
                fbeta_notbehavior.append(pr[2][0])

                # print performance metrics and feature importance to console
                print('-' * 70)
                print(f"training iteration {i}")
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
                print('-' * 70)
                print("Top 10 features by importance:")
                self._classifier.print_feature_importance(
                    features['column_names'],
                    10)

                # let the parent thread know that we've finished this iteration
                self._tasks_complete += 1
                self.update_progress.emit(self._tasks_complete)

            print('\n' + '=' * 70)
            print("SUMMARY\n")
            print(tabulate(table_rows, showindex="always", headers=[
                "accuracy", "precision\n(not behavior)",
                "precision\n(behavior)", "recall\n(not behavior)",
                "recall\n(behavior)", "f beta score\n(not behavior)",
                "f beta score\n(behavior)",
                "test - leave one out:\n(video [identity])"]))

            print(f"\nmean accuracy: {np.mean(accuracies):.5}")
            print(f"mean fbeta score (behavior): {np.mean(fbeta_behavior):.5}")
            print(f"std fbeta score (behavior): {np.std(fbeta_behavior):.05}")
            print(f"mean fbeta score (not behavior): {np.mean(fbeta_notbehavior):.5}")
            print(f"std fbeta score (not behavior): {np.std(fbeta_notbehavior):.05}")
            print(f"\nClassifier: {self._classifier.classifier_name}")
            print(f"Behavior: {self._behavior}")
            unit = "cm" if self._project.distance_unit == ProjectDistanceUnit.CM else "pixel"
            print(f"Feature Distance Unit: {unit}")
            print('-' * 70)

        # retrain with all training data and fixed random seed before saving:
        self.current_status.emit("Training and saving final classifier")
        self._classifier.train(
            {
                'training_data': self._classifier.combine_data(
                    features['per_frame'], features['window']),
                'training_labels': features['labels']
            },
            features['column_names'],
            self._behavior,
            self._window_size,
            self._uses_social,
            self._uses_balance,
            self._project.extended_features,
            self._project.distance_unit,
            random_seed=FINAL_TRAIN_SEED
        )

        self._project.save_classifier(self._classifier, self._behavior)
        self._tasks_complete += 1
        self.update_progress.emit(self._tasks_complete)

        self.current_status.emit("Training Complete")
        self.training_complete.emit()
