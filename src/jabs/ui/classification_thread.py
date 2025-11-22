import time
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QWidget

from jabs.classifier import Classifier
from jabs.feature_extraction import DEFAULT_WINDOW_SIZE, IdentityFeatures
from jabs.io import save_predictions
from jabs.project import Project

from ..pose_estimation import get_pose_path, open_pose_file
from .exceptions import ThreadTerminatedError


def classify_single_video_worker(
    video_name: str,
    pose_path: Path,
    classifier: "Classifier",
    project_settings: dict,
    feature_dir: Path,
    cache_dir: Path,
) -> dict:
    """Run classification for all identities in a single video.

    This helper is intentionally free of Qt and `Project` dependencies so it can
    be used in a ProcessPool-based parallelization strategy. It loads pose data
    from disk, computes features per identity, runs the classifier to obtain
    per-frame predictions and probabilities, and returns the results along with
    pose metadata.

    Since this is intended to be run in a separate process, all parameters are
    passed explicitly rather than relying on shared state and must
    be serializable by the multiprocessing module.

    Args:
        video_name: Name of the video file being processed.
        pose_path: Filesystem path to the corresponding pose file.
        classifier: Classifier instance used to compute probabilities.
        project_settings: Plain dictionary of behavior-specific settings.
        feature_dir: Directory where feature caches are stored.
        cache_dir: Directory used to cache pose estimation files.

    Returns:
        dict: A dictionary with keys:
            - "video": video name
            - "predictions": np.ndarray of shape (num_identities, num_frames)
            - "probabilities": np.ndarray of shape (num_identities, num_frames)
            - "frame_indexes": np.ndarray of shape (num_identities, num_frames)
            - "pose_file": basename of the pose file
            - "pose_hash": hash of the pose data
            - "identity_to_track": optional identity mapping array
            - "external_identities": optional list of external identity labels
    """
    # Load pose estimation for this video
    pose_est = open_pose_file(pose_path, cache_dir)
    fps = pose_est.fps
    num_identities = pose_est.num_identities
    num_frames = pose_est.num_frames

    # Allocate arrays for per-identity, per-frame outputs
    predictions = np.full((num_identities, num_frames), -1, dtype=np.int8)
    probabilities = np.zeros((num_identities, num_frames), dtype=np.float32)
    frame_indexes = np.zeros((num_identities, num_frames), dtype=np.int32)

    for identity in pose_est.identities:
        # get the features for this identity
        features = IdentityFeatures(
            video_name,
            identity,
            feature_dir,
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
        data = classifier.combine_data(per_frame_features, window_features)

        if data.shape[0] == 0:
            # leave default -1 / 0 rows as initialized
            continue

        # compute probabilities for all classes
        probs = classifier.predict_proba(data)

        # get predicted class for each frame using classifier's thresholding logic
        preds = classifier.threshold_probabilities(probs)

        idx = feature_values["frame_indexes"]
        predictions[identity, idx] = preds[idx]
        probabilities[identity, idx] = probs[np.arange(len(probs)), preds][idx]
        frame_indexes[identity, idx] = idx

    return {
        "video": video_name,
        "predictions": predictions,
        "probabilities": probabilities,
        "frame_indexes": frame_indexes,
        "pose_file": pose_est.pose_file.name,
        "pose_hash": pose_est.hash,
        "identity_to_track": pose_est.identity_to_track,
        "external_identities": pose_est.external_identities,
    }


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

    Note:
        A future enhancement could offload file-saving operations to a separate
        worker thread using Qt signals/slots. This would allow classification
        to continue without blocking on disk IO, improving throughput for
        projects with many videos.
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
        start_time = time.time()
        self._tasks_complete = 0
        current_video_predictions = {}
        current_video_probabilities = {}
        current_video_frame_indexes = {}

        def check_termination_requested() -> None:
            if self._should_terminate:
                raise ThreadTerminatedError("Classification was cancelled by the user")

        try:
            project_settings = self._project.settings_manager.get_behavior(self._behavior)
            project_settings_dict = dict(project_settings)
            cache_dir = self._project.project_paths.cache_dir
            feature_dir = self._project.feature_dir

            # iterate over each video in the project
            for video in self._project.video_manager.videos:
                check_termination_requested()

                video_path = self._project.video_manager.video_path(video)
                pose_path = get_pose_path(video_path)

                # high-level status for this video
                self.current_status.emit(f"Classifying {video}")

                # Run the per-video worker serially for now. This can later be
                # offloaded to a ProcessPoolExecutor without changing the
                # worker implementation.
                result = classify_single_video_worker(
                    video_name=video,
                    pose_path=pose_path,
                    classifier=self._classifier,
                    project_settings=project_settings_dict,
                    feature_dir=feature_dir,
                    cache_dir=cache_dir,
                )

                predictions = result["predictions"]
                probabilities = result["probabilities"]
                frame_indexes = result["frame_indexes"]
                pose_file = result["pose_file"]
                pose_hash = result["pose_hash"]
                identity_to_track = result["identity_to_track"]
                external_identities = result["external_identities"]

                num_identities = predictions.shape[0]

                if video == self._current_video:
                    current_video_predictions = {
                        identity: predictions[identity].copy()
                        for identity in range(num_identities)
                    }
                    current_video_probabilities = {
                        identity: probabilities[identity].copy()
                        for identity in range(num_identities)
                    }
                    current_video_frame_indexes = {
                        identity: frame_indexes[identity].copy()
                        for identity in range(num_identities)
                    }

                # save predictions to disk
                self.current_status.emit("Saving Predictions")
                output_path = (
                    self._project.project_paths.prediction_dir
                    / f"{Path(video).with_suffix('').name}.h5"
                )
                save_predictions(
                    output_path=output_path,
                    predictions=predictions,
                    probabilities=probabilities,
                    behavior=self._behavior,
                    classifier=self._classifier,
                    pose_file=pose_file,
                    pose_hash=pose_hash,
                    pose_identity_to_track=identity_to_track,
                    external_identities=external_identities,
                )

                self._tasks_complete += 1
                self.update_progress.emit(self._tasks_complete)

            # emit timing information
            elapsed = time.time() - start_time
            self.current_status.emit(f"Classification completed in {elapsed:.2f} seconds")
            # emits the predictions, probabilities, and frame indexes for the video currently loaded in
            # the video player, so that it can update the UI accordingly to show the new predictions
            self.classification_complete.emit(
                {
                    "predictions": current_video_predictions,
                    "probabilities": current_video_probabilities,
                    "frame_indexes": current_video_frame_indexes,
                }
            )
        except Exception as e:
            # if there was an exception, we'll emit the Exception as a signal so that
            # the main GUI thread can handle it
            self.error_callback.emit(e)
