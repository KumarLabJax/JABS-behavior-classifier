import time

import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from jabs.classifier import Classifier, MultiClassClassifier
from jabs.core.enums import ClassifierMode
from jabs.feature_extraction import DEFAULT_WINDOW_SIZE, IdentityFeatures
from jabs.project import Project

from .classify_strategy import (
    BinaryClassifyStrategy,
    ClassifyStrategy,
    MultiClassClassifyStrategy,
)
from .exceptions import ThreadTerminatedError


class ClassifyThread(QThread):
    """
    Thread used to run classification in the background, keeping the Qt main GUI thread responsive.

    Signals:
        classification_complete: QtCore.Signal(dict, int)
            Emitted when classification is finished successfully. The dict carries
            predictions, probabilities, post-processed predictions, and class_names
            for the current video so the UI can update; the int is the elapsed
            wall-clock time in milliseconds.
        current_status: QtCore.Signal(str)
            Emitted to update the main GUI thread with a status message (e.g., for a status bar).
        update_progress: QtCore.Signal(int)
            Emitted to inform the main GUI thread of the number of completed tasks
            (e.g., for a progress bar).
        error_callback: QtCore.Signal(Exception)
            Emitted if an error occurs during classification, passing the exception
            to the main GUI thread.

    Args:
        classifier (Classifier | MultiClassClassifier): The classifier to use for predictions.
        project (Project): The project containing data and settings.
        behavior (str): The behavior label to classify.
        current_video (str): The video currently loaded in the video player.
        videos (list[str] | None, optional): Restrict classification to these video
            filenames. When ``None`` (the default) every video in the project is
            classified, preserving the "classify all" behavior.
        parent (QWidget or None, optional): Optional parent widget.

    Note:
        A future enhancement could offload file-saving operations to a separate
        worker thread using Qt signals/slots. This would allow classification
        to continue without blocking on disk IO, improving throughput for
        projects with many videos.
    """

    classification_complete = Signal(dict, int)
    current_status = Signal(str)
    update_progress = Signal(int)
    error_callback = Signal(Exception)

    def __init__(
        self,
        classifier: Classifier | MultiClassClassifier,
        project: Project,
        behavior: str,
        current_video: str,
        videos: list[str] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self._classifier = classifier
        self._project = project
        self._behavior = behavior
        self._tasks_complete = 0
        self._current_video = current_video
        self._videos = videos
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

    def _build_strategy(self) -> ClassifyStrategy:
        """Construct the per-mode classification strategy for this run."""
        if self._project.settings_manager.classifier_mode == ClassifierMode.MULTICLASS:
            return MultiClassClassifyStrategy(
                classifier=self._classifier,
                project=self._project,
                behavior=self._behavior,
            )
        return BinaryClassifyStrategy(
            classifier=self._classifier,
            project=self._project,
            behavior=self._behavior,
        )

    def run(self) -> None:
        """Thread's main function.

        Runs the classifier for each identity in each video.
        """
        self._tasks_complete = 0
        current_video_predictions: dict = {}
        current_video_probabilities: dict = {}
        current_video_predictions_postprocessed: dict = {}
        t0_ns = time.perf_counter_ns()

        def check_termination_requested() -> None:
            if self._should_terminate:
                raise ThreadTerminatedError("Classification was cancelled by the user")

        try:
            strategy = self._build_strategy()
            project_settings = strategy.project_settings()

            videos = (
                self._videos if self._videos is not None else self._project.video_manager.videos
            )
            for video in videos:
                check_termination_requested()

                video_path = self._project.video_manager.video_path(video)
                pose_est = self._project.load_pose_est(video_path)
                fps = pose_est.fps

                predictions: dict = {}
                probabilities: dict = {}
                postprocessed_predictions: dict = {}

                for identity in pose_est.identities:
                    check_termination_requested()

                    self.current_status.emit(f"Classifying {video},  Identity {identity}")

                    features = IdentityFeatures(
                        video,
                        identity,
                        self._project.feature_dir,
                        pose_est,
                        fps=fps,
                        op_settings=project_settings,
                        cache_format=self._project.cache_format,
                    )
                    feature_values = features.get_features(
                        project_settings.get("window_size", DEFAULT_WINDOW_SIZE)
                    )

                    # reformat the data in a single 2D numpy array to pass to the classifier
                    per_frame_features = pd.DataFrame(feature_values["per_frame"])
                    window_features = pd.DataFrame(feature_values["window"])
                    data = self._classifier.combine_data(per_frame_features, window_features)

                    check_termination_requested()
                    if data.shape[0] > 0:
                        prob = self._classifier.predict_proba(
                            data, feature_values["frame_indexes"]
                        )
                        predictions[identity], confidence = self._classifier.derive_predictions(
                            prob
                        )
                        probabilities[identity] = strategy.probabilities_for_storage(
                            prob, confidence
                        )
                    else:
                        predictions[identity] = np.full(pose_est.num_frames, -1, dtype=np.int8)
                        probabilities[identity] = strategy.empty_probabilities(pose_est.num_frames)

                    postprocessed = strategy.postprocess_identity(
                        predictions[identity], probabilities[identity]
                    )
                    if postprocessed is not None:
                        postprocessed_predictions[identity] = postprocessed

                if video == self._current_video:
                    current_video_predictions = predictions
                    current_video_probabilities = probabilities
                    current_video_predictions_postprocessed = postprocessed_predictions

                self.current_status.emit("Saving Predictions")
                self._project.save_predictions(
                    pose_est,
                    video,
                    predictions,
                    probabilities,
                    strategy.prediction_behavior(),
                    self._classifier,
                    postprocessed_predictions=postprocessed_predictions,
                    class_names=strategy.class_names(),
                )

                self._tasks_complete += 1
                self.update_progress.emit(self._tasks_complete)

            elapsed_ms = int((time.perf_counter_ns() - t0_ns) // 1_000_000)
            self.classification_complete.emit(
                {
                    "predictions": current_video_predictions,
                    "probabilities": current_video_probabilities,
                    "predictions_postprocessed": current_video_predictions_postprocessed,
                    "class_names": strategy.class_names(),
                },
                elapsed_ms,
            )
        except Exception as e:
            self.error_callback.emit(e)
