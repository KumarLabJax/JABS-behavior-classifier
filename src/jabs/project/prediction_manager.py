import sys
import typing
from datetime import datetime
from pathlib import Path

import numpy as np

from jabs import io
from jabs.core.types.prediction import BehaviorPrediction, ClassifierMetadata
from jabs.version import version_str

if typing.TYPE_CHECKING:
    from jabs.classifier import Classifier
    from jabs.pose_estimation import PoseEstimation

    from .project import Project


class PredictionManager:
    """
    Manages reading and writing of prediction data for behaviors in a JABS project.

    Handles storage and retrieval of predicted classes and probabilities for each identity in each video,
    using HDF5 files. Provides methods to write new predictions, load existing predictions, and handle
    missing or invalid data. Integrates with the project structure to ensure predictions are associated
    with the correct behaviors and videos.

    Args:
        project: The JABS Project instance this manager is associated with.

    """

    _PREDICTION_FILE_VERSION = 2

    def __init__(self, project: "Project"):
        """Initialize the PredictionManager with a project.

        Args:
            project: JABS Project object
        """
        self._project = project

    @classmethod
    def write_predictions(
        cls,
        behavior: str,
        output_path: Path,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        poses: "PoseEstimation",
        classifier: "Classifier",
        postprocessed_predictions: np.ndarray | None = None,
    ) -> None:
        """
        Write predicted classes and probabilities for a behavior to an HDF5 file.

        Stores predictions, probabilities, and relevant metadata for each identity in the specified video.
        Optionally includes a mapping from JABS identities to external identities.

        Args:
            behavior (str): Name of the behavior for which predictions are made.
            output_path (Path): Path to the HDF5 file where predictions will be saved.
            predictions (np.ndarray): Array of predicted class labels, shape (n_animals, n_frames).
            probabilities (np.ndarray): Array of predicted class probabilities, shape (n_animals, n_frames).
            poses: PoseEstimation object corresponding to the video.
            classifier: Classifier object used to generate predictions.
            postprocessed_predictions (np.ndarray | None): Optional array of post-processed predictions.

        Returns:
            None
        """
        pred = BehaviorPrediction(
            behavior=behavior,
            predicted_class=predictions,
            probabilities=probabilities,
            classifier=ClassifierMetadata(
                classifier_file=classifier.classifier_file,
                classifier_hash=classifier.classifier_hash,
                app_version=version_str(),
                prediction_date=str(datetime.now()),
            ),
            pose_file=Path(poses.pose_file).name,
            pose_hash=poses.hash,
            predicted_class_postprocessed=postprocessed_predictions,
            identity_to_track=poses.identity_to_track,
            external_identity_mapping=poses.external_identities,
        )
        io.save(pred, output_path)

    def load_predictions(self, video: str, behavior: str):
        """load predictions for a given video and behavior

        Args:
            video: name of video to load predictions for
            behavior: behavior to load predictions for

        Returns:
            tuple of three dicts: (predictions, probabilities,
            predictions_postprocessed), where
        each dict has identities present in the video for keys
        """
        predictions = {}
        probabilities = {}
        postprocessed_predictions = {}

        file_base = Path(video).with_suffix("").name + ".h5"
        path = self._project.project_paths.prediction_dir / file_base

        nident = self._project.settings_manager.project_settings["video_files"][video][
            "identities"
        ]

        try:
            pred = io.load(path, BehaviorPrediction, behavior=behavior)
            assert pred.predicted_class.shape[0] == nident

            for i in range(nident):
                predictions[i] = pred.predicted_class[i]
                probabilities[i] = pred.probabilities[i]
                if pred.predicted_class_postprocessed is not None:
                    postprocessed_predictions[i] = pred.predicted_class_postprocessed[i]

        except (KeyError, FileNotFoundError):
            # no saved predictions for this behavior for this video
            pass
        except AssertionError:
            print(f"unable to open saved inferences for {video}", file=sys.stderr)

        return predictions, probabilities, postprocessed_predictions
