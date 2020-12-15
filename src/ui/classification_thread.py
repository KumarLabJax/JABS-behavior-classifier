import numpy as np
from PyQt5 import QtCore

from src.feature_extraction import IdentityFeatures


class ClassifyThread(QtCore.QThread):
    """
    thread to run the classification to keep the main GUI thread responsive
    """

    done = QtCore.pyqtSignal()
    update_progress = QtCore.pyqtSignal(int)
    current_status = QtCore.pyqtSignal(str)

    def __init__(self, classifier, project, behavior, predictions,
                 probabilities, frame_indexes, current_video):
        super().__init__()
        self._classifier = classifier
        self._project = project
        self._behavior = behavior
        self._predictions = predictions
        self._probabilities = probabilities
        self._frame_indexes = frame_indexes
        self._tasks_complete = 0
        self._current_video = current_video

    def run(self):
        """
        thread's main function. runs the classifier for each identity in each
        video
        """
        self._tasks_complete = 0

        predictions = {}
        probabilities = {}
        frame_indexes = {}

        # iterate over each video in the project
        for video in self._project.videos:

            # load the poses for this video
            pose_est = self._project.load_pose_est(
                self._project.video_path(video))

            # make predictions for each identity in this video
            predictions[video] = {}
            probabilities[video] = {}
            frame_indexes[video] = {}

            for ident in pose_est.identities:
                self.current_status.emit(f"Classifying {video} identity={ident}")

                # get the features for this identity
                features = IdentityFeatures(video, ident,
                                            self._project.feature_dir,
                                            pose_est)
                identity = str(ident)

                labels = self._project.load_video_labels(
                    video, leave_cached=True
                ).get_track_labels(identity, self._behavior).get_labels()

                # get the features for all unlabled frames for this identity
                # TODO make window_size configurable
                unlabeled_features = features.get_unlabeled_features(5, labels)

                # reformat the data in a single 2D numpy array to pass
                # to the classifier
                data = self._classifier.combine_data(
                    unlabeled_features['per_frame'],
                    unlabeled_features['window']
                )

                # make predictions
                predictions[video][identity] = self._classifier.predict(
                    data)

                # also get the probabilities
                prob = self._classifier.predict_proba(data)
                # Save the probability for the predicted class only.
                # The following code uses some
                # numpy magic to use the _predictions array as column indexes
                # for each row of the 'prob' array we just computed.
                probabilities[video][identity] = prob[
                    np.arange(len(prob)),
                    predictions[video][identity]
                ]

                # save the indexes for the predicted frames
                frame_indexes[video][identity] = unlabeled_features[
                    'frame_indexes']
                self._tasks_complete += 1
                self.update_progress.emit(self._tasks_complete)

        # save predictions
        self.current_status.emit("Saving Predictions")
        self._project.save_predictions(predictions,
                                       probabilities,
                                       frame_indexes,
                                       self._behavior)

        self._predictions = predictions[self._current_video]
        self._probabilities = probabilities[self._current_video]
        self._frame_indexes = frame_indexes[self._current_video]
        self.done.emit()
