import numpy as np
from PyQt5 import QtCore

from src.labeler import TrackLabels
from src.feature_extraction import IdentityFeatures


class TrainingThread(QtCore.QThread):
    """
    Thread used to run the training to keep the Qt main GUI thread responsive.
    """

    trainingComplete = QtCore.pyqtSignal()
    currentStatus = QtCore.pyqtSignal(str)

    def __init__(self, project, classifier, behavior, current_video,
                 current_labels):
        QtCore.QThread.__init__(self)
        self._project = project
        self._classifier = classifier
        self._behavior = behavior
        self._current_video = current_video
        self._current_labels = current_labels

    def run(self):
        """
        thread's main function. Will get the feature set for all labeled frames,
        do the leave one group out train/test split, run the training, run the
        trained classifier on the test data, print the accuracy on the test
        data, and print the most important features
        """

        features = self._get_labeled_features()

        data = self._classifier.leave_one_group_out(
            features['per_frame'],
            features['window'],
            features['labels'],
            features['groups']
        )

        self._classifier.train(data)
        predictions = self._classifier.predict(data['test_data'])

        correct = 0
        for p, truth in zip(predictions, data['test_labels']):
            if p == truth:
                correct += 1
        print(f"accuracy: {correct / len(predictions) * 100:.2f}%")

        self._classifier.print_feature_importance(
            IdentityFeatures.get_feature_names())

        # let the parent thread know that we've finished
        self.trainingComplete.emit()

    def _get_labeled_features(self):

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
                        video).get_track_labels(str(identity),
                                                self._behavior).get_labels()

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

        return {
            'window': IdentityFeatures.merge_window_features(all_window),
            'per_frame': IdentityFeatures.merge_per_frame_features(
                all_per_frame),
            'labels': np.concatenate(all_labels),
            'groups': np.concatenate(all_group_labels),
        }
