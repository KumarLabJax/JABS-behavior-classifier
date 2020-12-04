import itertools

import numpy as np
from PyQt5 import QtCore
from tabulate import tabulate

from src.feature_extraction import IdentityFeatures


class TrainingThread(QtCore.QThread):
    """
    Thread used to run the training to keep the Qt main GUI thread responsive.
    """

    # signal so that the main GUI thread can be notified when the training is
    # complete
    trainingComplete = QtCore.pyqtSignal()

    # allow the thread to send a status string to the main GUI thread so that
    # we can update a status bar if we want
    currentStatus = QtCore.pyqtSignal(str)

    update_progress = QtCore.pyqtSignal(int)

    def __init__(self, project, classifier, behavior, k=1):
        super().__init__()
        self._project = project
        self._classifier = classifier
        self._behavior = behavior
        self._tasks_complete = 0
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
        features, group_mapping = self._project.get_labeled_features(
            self._behavior,
            id_processed,
        )

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

        for i, data in enumerate(itertools.islice(data_generator, self._k)):

            test_info = group_mapping[data['test_group']]

            # train classifier, and then use it to classify our test data
            self._classifier.train(data)
            predictions = self._classifier.predict(data['test_data'])

            # calculate some performance metrics using the classifications of the
            # test data
            accuracy = self._classifier.accuracy_score(data['test_labels'],
                                                       predictions)
            pr = self._classifier.precision_recall_score(data['test_labels'],
                                                         predictions)
            confusion = self._classifier.confusion_matrix(data['test_labels'],
                                                          predictions)

            table_rows.append([accuracy, pr[0][0], pr[0][1], pr[1][0], pr[1][1],
                               pr[2][0], pr[2][1],
                               f"{test_info['video']} [{test_info['identity']}]"])
            accuracies.append(accuracy)
            fbeta_behavior.append(pr[2][0])
            fbeta_notbehavior.append(pr[2][1])


            # print performance metrics and feature importance to console
            print('-' * 70)
            print(f"training iteration {i}")
            print("TEST DATA:")
            print(f"\tVideo: {test_info['video']}")
            print(f"\tIdentity: {test_info['identity']}")
            print(f"ACCURACY: {accuracy * 100:.2f}%")
            print("PRECISION RECALL:")
            print(f"              {'behavior':12}  not behavior")
            print(f"  precision   {pr[0][0]:<12.8}  {pr[0][1]:<.8}")
            print(f"  recall      {pr[1][0]:<12.8}  {pr[1][1]:<.8}")
            print(f"  fbeta score {pr[2][0]:<12.8}  {pr[2][1]:<.8}")
            print(f"  support     {pr[3][0]:<12}  {pr[3][1]}")
            print("CONFUSION MATRIX:")
            print(f"{confusion}")
            print('-' * 70)
            print("Top 10 features by importance:")
            self._classifier.print_feature_importance(
                IdentityFeatures.get_feature_names(
                    self._project.has_social_features),
                10)

            # let the parent thread know that we've finished
            self._tasks_complete += 1
            self.update_progress.emit(self._tasks_complete)

        self._project.save_classifier(self._classifier, self._behavior)

        print('\n' + '=' * 70)
        print("SUMMARY\n")
        print(tabulate(table_rows, showindex="always", headers=[
            "accuracy", "precision\n(behavior)",
            "precision\n(not behavior)", "recall\n(behavior)",
            "recall\n(not behavior)", "f beta score\n(behavior)",
            "f beta score\n(not behavior)",
            "test - leave one out:\n(video [identity])"]))

        print(f"\nmean accuracy: {np.mean(accuracies):.5}")
        print(f"mean fbeta score (behavior): {np.mean(fbeta_behavior):.5}")
        print("mean fbeta score (not behavior): "
              f"{np.mean(fbeta_notbehavior):.5}")
        print(f"Classifier: {self._classifier.classifier_name}")
        print('-' * 70)

        self.trainingComplete.emit()
