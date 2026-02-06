import sys
import typing
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from jabs.core.exceptions import MissingBehaviorError
from jabs.version import version_str

from .project_utils import to_safe_name

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
        # TODO catch exceptions
        with h5py.File(output_path, "a") as h5:
            h5.attrs["pose_file"] = Path(poses.pose_file).name
            h5.attrs["pose_hash"] = poses.hash
            h5.attrs["version"] = cls._PREDICTION_FILE_VERSION
            prediction_group = h5.require_group("predictions")
            # Write external identity mapping only if not already present.
            if (
                poses.external_identities is not None
                and "external_identity_mapping" not in prediction_group
            ):
                prediction_group.create_dataset(
                    "external_identity_mapping",
                    data=np.array(poses.external_identities, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            behavior_group = prediction_group.require_group(to_safe_name(behavior))
            behavior_group.attrs["classifier_file"] = classifier.classifier_file
            behavior_group.attrs["classifier_hash"] = classifier.classifier_hash
            behavior_group.attrs["app_version"] = version_str()
            behavior_group.attrs["prediction_date"] = str(datetime.now())
            h5_predictions = behavior_group.require_dataset(
                "predicted_class", shape=predictions.shape, dtype=predictions.dtype
            )
            h5_predictions[...] = predictions

            h5_probabilities = behavior_group.require_dataset(
                "probabilities", shape=probabilities.shape, dtype=probabilities.dtype
            )
            h5_probabilities[...] = probabilities

            if postprocessed_predictions is not None:
                h5_postprocessed = behavior_group.require_dataset(
                    "predicted_class_postprocessed",
                    shape=postprocessed_predictions.shape,
                    dtype=postprocessed_predictions.dtype,
                )
                h5_postprocessed[...] = postprocessed_predictions

            if poses.identity_to_track is not None:
                h5_ids = behavior_group.require_dataset(
                    "identity_to_track",
                    shape=poses.identity_to_track.shape,
                    dtype=poses.identity_to_track.dtype,
                )
                h5_ids[...] = poses.identity_to_track
            elif "identity_to_track" in behavior_group:
                del behavior_group["identity_to_track"]

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
            with h5py.File(path, "r") as h5:
                assert h5.attrs["version"] == self._PREDICTION_FILE_VERSION
                prediction_group = h5["predictions"]
                if to_safe_name(behavior) not in prediction_group:
                    # This needs to appear as if no saved predictions exist for this video.
                    raise MissingBehaviorError(
                        f"Behavior {to_safe_name(behavior)} not in prediction file."
                    )
                behavior_group = prediction_group[to_safe_name(behavior)]
                assert behavior_group["predicted_class"].shape[0] == nident
                assert behavior_group["probabilities"].shape[0] == nident

                _probabilities = behavior_group["probabilities"][:]
                _classes = behavior_group["predicted_class"][:]

                if "predicted_class_postprocessed" in behavior_group:
                    _postprocessed = behavior_group["predicted_class_postprocessed"][:]
                else:
                    _postprocessed = None

                for i in range(nident):
                    predictions[i] = _classes[i]
                    probabilities[i] = _probabilities[i]
                    if _postprocessed is not None:
                        postprocessed_predictions[i] = _postprocessed[i]

        except (MissingBehaviorError, FileNotFoundError):
            # no saved predictions for this behavior for this video
            pass
        except (AssertionError, KeyError):
            print(f"unable to open saved inferences for {video}", file=sys.stderr)

        return predictions, probabilities, postprocessed_predictions
