import numpy as np
from PyQt5 import QtCore

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
                 current_labels):
        QtCore.QThread.__init__(self)
        self._project = project
        self._classifier = classifier
        self._behavior = behavior
        self._current_video = current_video
        self._current_labels = current_labels
        self._tasks_complete = 0

    def run(self):
        """
        thread's main function. Will get the feature set for all labeled frames,
        do the leave one group out train/test split, run the training, run the
        trained classifier on the test data, print some performance metrics,
        and print the most important features
        """

        self._tasks_complete = 0
        features = self._get_labeled_features()

        data = self._classifier.leave_one_group_out(
            features['per_frame'],
            features['window'],
            features['labels'],
            features['groups']
        )

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

        # print performance metrics and feature importance to console
        print('-' * 70)
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
            IdentityFeatures.get_feature_names(), 10)

        # let the parent thread know that we've finished
        self._tasks_complete += 1
        self.update_progress.emit(self._tasks_complete)
        self.trainingComplete.emit()

    def _get_labeled_features(self):
        """
        the the features for all labeled frames
        NOTE: this will currently take a very long time to run if the features
        have not already been computed
        :return: dict with the following keys:
        {
            'window': ,
            'per_frame': ,
            'labels': ,
            'groups': ,
        }
        The values of these are suitable to pass as arguments to the
        SklClassifier.leave_one_group_out() method
        """

        all_per_frame = []
        all_window = []
        all_labels = []
        all_group_labels = []

        group_id = 0
        for video in self._project.videos:

            pose_est = self._project.load_pose_est(
                self._project.video_path(video))

            for identity in pose_est.identities:
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
        }
