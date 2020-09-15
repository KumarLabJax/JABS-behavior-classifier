import numpy as np
from PyQt5 import QtCore

from src.feature_extraction import IdentityFeatures


class ClassifyThread(QtCore.QThread):
    """
    thread to run the classification to keep the main GUI thread responsive
    """

    done = QtCore.pyqtSignal()
    update_progress = QtCore.pyqtSignal(int)

    def __init__(self, classifier, project, behavior, current_video,
                 current_labels, predictions, probabilities, frame_indexes):
        QtCore.QThread.__init__(self)
        self._classifier = classifier
        self._project = project
        self._behavior = behavior
        self._current_video = current_video
        self._current_labels = current_labels
        self._predictions = predictions
        self._probabilities = probabilities
        self._frame_indexes = frame_indexes
        self._tasks_complete = 0

    def run(self):
        """
        thread's main function. runs the classifier for each identity in each
        video
        TODO: could use more multi-threading speed up
        """
        self._tasks_complete = 0
        # iterate over each video in the project
        for video in self._project.videos:

            # load the poses for this video
            pose_est = self._project.load_pose_est(
                self._project.video_path(video))

            # make predictions for each identity in this video
            self._predictions[video] = {}
            self._probabilities[video] = {}
            self._frame_indexes[video] = {}

            for ident in pose_est.identities:
                # get the features for this identity
                features = IdentityFeatures(video, ident,
                                            self._project.feature_dir,
                                            pose_est)
                identity = str(ident)

                if self._project.video_path(video) == self._current_video:
                    # if this is the current video, the labels are loaded
                    labels = self._current_labels.get_track_labels(
                        identity, self._behavior).get_labels()
                else:
                    # all other videos, load the labels from the project dir
                    labels = self._project.load_annotation_track(
                        video).get_track_labels(identity, self._behavior).get_labels()

                # get the features for all unlabled frames for this identity
                # TODO make window radius configurable
                unlabeled_features = features.get_unlabeled_features(5, labels)

                # reformat the data in a single 2D numpy array to pass
                # to the classifier
                data = self._classifier.combine_data(
                    unlabeled_features['per_frame'],
                    unlabeled_features['window']
                )

                # make predictions
                self._predictions[video][identity] = self._classifier.predict(
                    data)

                # also get the probabilities
                prob = self._classifier.predict_proba(data)
                # Save the probability for the predicted class only.
                # self._predictions[video][identity] will be an array of
                # 1s and 2s, since those are our labels. Subtracting 1 from the
                # predicted label will give us the column index for the
                # probability for that label. The following code uses some
                # numpy magic to use the _predictions array as column indexes
                # for each row of the 'prob' array we just computed.
                self._probabilities[video][identity] = prob[
                    np.arange(len(prob)),
                    self._predictions[video][identity] - 1
                ]

                # save the indexes for the predicted frames
                self._frame_indexes[video][identity] = unlabeled_features[
                    'frame_indexes']
                self._tasks_complete += 1
                self.update_progress.emit(self._tasks_complete)
        self.done.emit()