import sys
import typing
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .project_utils import to_safe_name
from jabs.version import version_str

if typing.TYPE_CHECKING:
    from .project import Project


class MissingBehaviorError(Exception):
    pass

class PredictionManager:
    """
    Class to manage the loading and saving of predictions.
    """

    _PREDICTION_FILE_VERSION = 2

    def __init__(self, project: "Project"):
        """
        Initialize the PredictionManager with a project.

        :param project: JABS Project object
        """
        self._project = project

    @classmethod
    def write_predictions(
        cls,
        behavior: str,
        output_path: Path,
        predictions,
        probabilities,
        poses,
        classifier,
    ):
        """
        write predictions out to a file
        :param behavior: string describing the behavior
        :param output_path: name of file to write predictions to
        :param predictions: matrix of prediction class data of shape [n_animals, n_frames]
        :param probabilities: matrix of probability for the predicted class of shape [n_animals, n_frames]
        :param poses: PoseEstimation object for which predictions were made
        :param classifier: Classifier object for which was used to make predictions
        """
        # TODO catch exceptions
        with h5py.File(output_path, "a") as h5:
            h5.attrs["pose_file"] = Path(poses.pose_file).name
            h5.attrs["pose_hash"] = poses.hash
            h5.attrs["version"] = cls._PREDICTION_FILE_VERSION
            prediction_group = h5.require_group("predictions")
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
        """
        load predictions for a given video and behavior
        :param video: name of video to load predictions for
        :param behavior: behavior to load predictions for
        :return: tuple of three dicts: (predictions, probabilities, frame_indexes)
        each dict has identities present in the video for keys
        """

        predictions = {}
        probabilities = {}
        frame_indexes = {}

        file_base = Path(video).with_suffix("").name + ".h5"
        path = self._project.project_paths.prediction_dir / file_base

        nident = self._project.settings_manager.project_settings["video_files"][video]["identities"]

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

                for i in range(nident):
                    identity = str(i)
                    indexes = np.asarray(
                        range(behavior_group["predicted_class"].shape[1])
                    )

                    # first, exclude any probability of -1 as that indicates
                    # a user label, not an inferred class
                    indexes = indexes[_probabilities[i] != -1]

                    # now excludes a class of -1 as that indicates the
                    # identity isn't present
                    indexes = indexes[_classes[i, indexes] != -1]

                    # we're left with classes/probabilities for frames that
                    # were inferred and their frame indexes
                    predictions[identity] = _classes[i]
                    probabilities[identity] = _probabilities[i]
                    frame_indexes[identity] = indexes

        except MissingBehaviorError:
            # no saved predictions for this behavior for this video
            pass
        except (AssertionError, KeyError) as e:
            print(f"unable to open saved inferences for {video}", file=sys.stderr)

        return predictions, probabilities, frame_indexes
