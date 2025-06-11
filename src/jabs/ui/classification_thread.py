import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal, SignalInstance

from jabs.feature_extraction import DEFAULT_WINDOW_SIZE, IdentityFeatures
from jabs.video_reader.utilities import get_fps


class ClassifyThread(QThread):
    """thread to run the classification to keep the main GUI thread responsive"""

    # signal so that the main GUI thread can be notified when classification is complete
    classification_complete: SignalInstance = Signal(dict)

    # allow the thread to send a status string to the main GUI thread so that
    # we can update a status bar if we want
    current_status: SignalInstance = Signal(str)

    # signal to inform the main GUI thread of the number of tasks completed
    # so that it can update a progress bar
    update_progress: SignalInstance = Signal(int)

    # inform the main GUI thread if there was an error during training
    error_callback: SignalInstance = Signal(Exception)

    def __init__(self, classifier, project, behavior, current_video, parent=None):
        super().__init__(parent=parent)
        self._classifier = classifier
        self._project = project
        self._behavior = behavior
        self._tasks_complete = 0
        self._current_video = current_video

    def run(self):
        """thread's main function.

        runs the classifier for each identity in each video
        """
        self._tasks_complete = 0

        predictions = {}
        probabilities = {}
        frame_indexes = {}

        try:
            project_settings = self._project.settings_manager.get_behavior(self._behavior)

            # iterate over each video in the project
            for video in self._project.video_manager.videos:
                video_path = self._project.video_manager.video_path(video)

                # load the poses for this video
                pose_est = self._project.load_pose_est(video_path)
                # fps used to scale some features from per pixel time unit to
                # per second
                fps = get_fps(str(video_path))

                # make predictions for each identity in this video
                predictions[video] = {}
                probabilities[video] = {}
                frame_indexes[video] = {}

                for identity in pose_est.identities:
                    self.current_status.emit(f"Classifying {video},  Identity {identity}")

                    # get the features for this identity
                    features = IdentityFeatures(
                        video,
                        identity,
                        self._project.feature_dir,
                        pose_est,
                        fps=fps,
                        op_settings=project_settings,
                    )
                    feature_values = features.get_features(
                        project_settings.get("window_size", DEFAULT_WINDOW_SIZE)
                    )

                    # reformat the data in a single 2D numpy array to pass
                    # to the classifier
                    per_frame_features = pd.DataFrame(
                        IdentityFeatures.merge_per_frame_features(feature_values["per_frame"])
                    )
                    window_features = pd.DataFrame(
                        IdentityFeatures.merge_window_features(feature_values["window"])
                    )
                    data = self._classifier.combine_data(per_frame_features, window_features)

                    if data.shape[0] > 0:
                        # make predictions
                        predictions[video][identity] = self._classifier.predict(data)

                        # also get the probabilities
                        prob = self._classifier.predict_proba(data)
                        # Save the probability for the predicted class only.
                        # The following code uses some
                        # numpy magic to use the _predictions array as column indexes
                        # for each row of the 'prob' array we just computed.
                        probabilities[video][identity] = prob[
                            np.arange(len(prob)), predictions[video][identity]
                        ]

                        # save the indexes for the predicted frames
                        frame_indexes[video][identity] = feature_values["frame_indexes"]
                    else:
                        predictions[video][identity] = np.array(0)
                        probabilities[video][identity] = np.array(0)
                        frame_indexes[video][identity] = np.array(0)
                    self._tasks_complete += 1
                    self.update_progress.emit(self._tasks_complete)

            # save predictions
            self.current_status.emit("Saving Predictions")
            self._project.save_predictions(
                predictions, probabilities, frame_indexes, self._behavior, self._classifier
            )

            self._tasks_complete += 1
            self.update_progress.emit(self._tasks_complete)
            self.classification_complete.emit(
                {
                    "predictions": predictions[self._current_video],
                    "probabilities": probabilities[self._current_video],
                    "frame_indexes": frame_indexes[self._current_video],
                }
            )
        except Exception as e:
            # if there was an exception, we'll emit the Exception as a signal so that
            # the main GUI thread can handle it
            self.error_callback.emit(e)
