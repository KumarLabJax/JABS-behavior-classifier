import itertools

import numpy as np
from PyQt5 import QtCore
from tabulate import tabulate

from src.feature_extraction import IdentityFeatures
from src.labeler import TrackLabels


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

    def __init__(self, project, classifier, behavior, current_video,
                 current_labels, k=1):
        super().__init__()
        self._project = project
        self._classifier = classifier
        self._behavior = behavior
        self._current_video = current_video
        self._current_labels = current_labels
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
        features, group_mapping = self._get_labeled_features()

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

        for data, i in zip(itertools.islice(data_generator, self._k),
                           range(self._k)):

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
        print('-' * 70)

        self.trainingComplete.emit()

    def _get_labeled_features(self):
        """
        the the features for all labeled frames
        NOTE: this will currently take a very long time to run if the features
        have not already been computed

        :return: two dicts: features, group_mappings

        The first dict contains features for all labeled frames and has the
        following keys:

        {
            'window': ,
            'per_frame': ,
            'labels': ,
            'groups': ,
        }

        The values contained in the first dict are suitable to pass as
        arguments to the SklClassifier.leave_one_group_out() method.

        The second dict in the tuple has group ids as the keys, and the
        values are a dict containing the video and identity that corresponds to
        that group id:

        {
          <group id>: {'video': <video filename>, 'identity': <identity},
          ...
        }
        """

        all_per_frame = []
        all_window = []
        all_labels = []
        all_group_labels = []

        group_mapping = {}

        group_id = 0
        for video in self._project.videos:

            pose_est = self._project.load_pose_est(
                self._project.video_path(video))

            for identity in pose_est.identities:
                group_mapping[group_id] = {'video': video, 'identity': identity}

                features = IdentityFeatures(video, identity,
                                            self._project.feature_dir,
                                            pose_est)

                if self._project.video_path(video) == self._current_video:
                    labels = self._current_labels.get_track_labels(
                        str(identity), self._behavior).get_labels()
                else:
                    labels = self._project.load_annotation_track(
                        video, leave_cached=True
                    ).get_track_labels(str(identity), self._behavior).get_labels()

                per_frame_features = features.get_per_frame(labels)
                # TODO make window size configurable
                window_features = features.get_window_features(5, labels)

                all_per_frame.append(per_frame_features)
                all_window.append(window_features)
                all_labels.append(labels[labels != TrackLabels.Label.NONE])

                # should be a better way to do this, but I'm getting the number
                # of frames in this group by looking at the shape of one of
                # the arrays included in the window_features
                all_group_labels.append(
                    np.full(window_features['percent_frames_present'].shape[0],
                            group_id))
                group_id += 1

                self._tasks_complete += 1
                self.update_progress.emit(self._tasks_complete)

        return {
            'window': IdentityFeatures.merge_window_features(all_window),
            'per_frame': IdentityFeatures.merge_per_frame_features(
                all_per_frame),
            'labels': np.concatenate(all_labels),
            'groups': np.concatenate(all_group_labels),
        }, group_mapping
