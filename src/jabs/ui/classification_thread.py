import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from jabs.classifier import Classifier
from jabs.feature_extraction import DEFAULT_WINDOW_SIZE, IdentityFeatures
from jabs.project import Project

from .exceptions import ThreadTerminatedError


class ClassifyThread(QThread):
    """
    Thread used to run classification in the background, keeping the Qt main GUI thread responsive.

    Signals:
        classification_complete: QtCore.Signal(dict)
            Emitted when classification is finished successfully. The emitted dict
            contains predictions, probabilities, and frame indexes for the current video so that
            the UI can update accordingly.
        current_status: QtCore.Signal(str)
            Emitted to update the main GUI thread with a status message (e.g., for a status bar).
        update_progress: QtCore.Signal(int)
            Emitted to inform the main GUI thread of the number of completed tasks
            (e.g., for a progress bar).
        error_callback: QtCore.Signal(Exception)
            Emitted if an error occurs during classification, passing the exception
            to the main GUI thread.

    Args:
        classifier (Classifier): The classifier instance to use for predictions.
        project (Project): The project containing data and settings.
        behavior (str): The behavior label to classify.
        current_video (str): The video currently loaded in the video player.
        parent (QWidget or None, optional): Optional parent widget.
    """

    classification_complete = Signal(dict)
    current_status = Signal(str)
    update_progress = Signal(int)
    error_callback = Signal(Exception)

    def __init__(
        self,
        classifier: Classifier,
        project: Project,
        behavior: str,
        current_video: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self._classifier = classifier
        self._project = project
        self._behavior = behavior
        self._tasks_complete = 0
        self._current_video = current_video
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
        """thread's main function.

        runs the classifier for each identity in each video
        """
        self._tasks_complete = 0

        predictions = {}
        probabilities = {}
        frame_indexes = {}

        def check_termination_requested() -> None:
            if self._should_terminate:
                raise ThreadTerminatedError("Classification was cancelled by the user")

        try:
            project_settings = self._project.settings_manager.get_behavior(self._behavior)

            # iterate over each video in the project
            for video in self._project.video_manager.videos:
                check_termination_requested()

                video_path = self._project.video_manager.video_path(video)
                pose_est = self._project.load_pose_est(video_path)
                fps = pose_est.fps

                # make predictions for each identity in this video
                predictions[video] = {}
                probabilities[video] = {}
                frame_indexes[video] = {}

                for identity in pose_est.identities:
                    check_termination_requested()

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

                    # reformat the data in a single 2D numpy array to pass to the classifier
                    per_frame_features = pd.DataFrame(
                        IdentityFeatures.merge_per_frame_features(feature_values["per_frame"])
                    )
                    window_features = pd.DataFrame(
                        IdentityFeatures.merge_window_features(feature_values["window"])
                    )
                    data = self._classifier.combine_data(per_frame_features, window_features)

                    check_termination_requested()
                    if data.shape[0] > 0:
                        # make predictions
                        # Note: this makes predictions for all frames in the video, even those without valid pose
                        # We will later filter these out when saving the predictions to disk
                        # consider changing this to only predict on frames with valid pose
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

            # save predictions to disk
            self.current_status.emit("Saving Predictions")
            self._project.save_predictions(
                predictions, probabilities, frame_indexes, self._behavior, self._classifier
            )

            self._tasks_complete += 1
            self.update_progress.emit(self._tasks_complete)

            # emits the predictions, probabilities, and frame indexes for the video currently loaded in
            # the video player, so that it can update the UI accordingly to show the new predictions
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
