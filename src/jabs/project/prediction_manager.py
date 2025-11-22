import sys
import typing
from pathlib import Path

import h5py
import numpy as np

from .project_utils import to_safe_name

if typing.TYPE_CHECKING:
    from .project import Project


class MissingBehaviorError(Exception):
    """Exception raised when a behavior is not found in the prediction file."""

    pass


class PredictionManager:
    """
    Manages reading and writing of prediction data for behaviors in a JABS project.

    Handles storage and retrieval of predicted classes and probabilities for each identity in each video,
    using HDF5 files. Provides methods to write new predictions, load existing predictions, and handle
    missing or invalid data. Integrates with the project structure to ensure predictions are associated
    with the correct behaviors and videos.

    Args:
        project: The JABS Project instance this manager is associated with.

    Note:
        This should be deprecated in favor of standalone functions in the jabs.io package.
          save_predictions has been migrated; load_predictions is pending migration to jabs.io.
    """

    _PREDICTION_FILE_VERSION = 2

    def __init__(self, project: "Project"):
        """Initialize the PredictionManager with a project.

        Args:
            project: JABS Project object
        """
        self._project = project

    def load_predictions(self, video: str, behavior: str):
        """load predictions for a given video and behavior

        Args:
            video: name of video to load predictions for
            behavior: behavior to load predictions for

        Returns:
            tuple of three dicts: (predictions, probabilities,
            frame_indexes)
        each dict has identities present in the video for keys
        """
        predictions = {}
        probabilities = {}
        frame_indexes = {}

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

                for i in range(nident):
                    indexes = np.asarray(range(behavior_group["predicted_class"].shape[1]))

                    # first, exclude any probability of -1 as that indicates
                    # a user label, not an inferred class
                    indexes = indexes[_probabilities[i] != -1]

                    # now excludes a class of -1 as that indicates the
                    # identity isn't present
                    indexes = indexes[_classes[i, indexes] != -1]

                    # we're left with classes/probabilities for frames that
                    # were inferred and their frame indexes
                    predictions[i] = _classes[i]
                    probabilities[i] = _probabilities[i]
                    frame_indexes[i] = indexes

        except (MissingBehaviorError, FileNotFoundError):
            # no saved predictions for this behavior for this video
            pass
        except (AssertionError, KeyError):
            print(f"unable to open saved inferences for {video}", file=sys.stderr)

        return predictions, probabilities, frame_indexes
