from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from jabs.project.project_utils import to_safe_name
from jabs.version import version_str

if TYPE_CHECKING:
    from jabs.classifier import Classifier


PREDICTION_FILE_VERSION = 2


def save_predictions(
    *,
    output_path: Path,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    behavior: str,
    classifier: "Classifier",
    pose_file: str,
    pose_hash: str,
    pose_identity_to_track: np.ndarray | None,
    external_identities: list[str] | None,
) -> None:
    """
    Saves classifier outputs for a single video/behavior to an HDF5 predictions file.

    This function writes (or updates) the prediction file at `output_path` using a
    fixed internal structure:

        /predictions/
            external_identity_mapping      (optional)
            <behavior-safe-name>/
                predicted_class            (int array, shape [num_ids, num_frames])
                probabilities              (float array, shape [num_ids, num_frames])
                identity_to_track          (optional: array mapping pose identities)
                classifier_file            (attribute)
                classifier_hash            (attribute)
                app_version                (attribute)
                prediction_date            (attribute)

    The function will:
    * Create the file if it does not exist, or open it in append/update mode.
    * Add an external identity mapping if provided and not already present.
    * Store predictions and probabilities as datasets under the behavior-specific
      group.
    * Embed metadata describing the classifier, application version, pose file,
      and pose hash.
    * Update or remove the identity-to-track dataset depending on the provided
      value.

    Args:
        output_path (Path): Path to the HDF5 file where predictions will be saved.
        predictions (np.ndarray): Integer array of shape
            `(num_identities, num_frames)` containing the predicted class index per
            frame for each identity.
        probabilities (np.ndarray): Float array of shape
            `(num_identities, num_frames)` containing the probability of the
            predicted class for each frame.
        behavior (str): Name of the behavior being classified; used as a group
            name inside the prediction file.
        classifier (Classifier): Classifier used to generate predictions. Metadata
            such as classifier filename and hash are stored for reproducibility.
        pose_file (str): Path to the pose file used to compute features.
        pose_hash (str): Hash of the pose data, used to detect stale or
            incompatible predictions.
        pose_identity_to_track (np.ndarray | None): Optional array mapping pose
            identities to tracked identities. If provided, the dataset is created or
            updated. If `None`, any existing dataset is removed.
        external_identities (list[str] | None): Optional list mapping identity index
            to external identity name. When provided, this mapping is stored once
            under `/predictions/`.

    Raises:
        ValueError: If `external_identities` is provided but its length does not
            match `predictions.shape[0]`.

    Returns:
        None: This function writes data to disk and does not return a value.
    """
    if external_identities and predictions.shape[0] != len(external_identities):
        raise ValueError(
            f"Predictions shape[0] {predictions.shape} does not match number of external identities {len(external_identities)}"
        )

    with h5py.File(output_path, "a") as h5:
        h5.attrs["pose_file"] = pose_file
        h5.attrs["pose_hash"] = pose_hash
        h5.attrs["version"] = PREDICTION_FILE_VERSION
        prediction_group = h5.require_group("predictions")
        # Write external identity mapping only if not already present.
        if external_identities is not None and "external_identity_mapping" not in prediction_group:
            prediction_group.create_dataset(
                "external_identity_mapping",
                data=np.array(external_identities, dtype=object),
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
        if pose_identity_to_track is not None:
            h5_ids = behavior_group.require_dataset(
                "identity_to_track",
                shape=pose_identity_to_track.shape,
                dtype=pose_identity_to_track.dtype,
            )
            h5_ids[...] = pose_identity_to_track
        elif "identity_to_track" in behavior_group:
            del behavior_group["identity_to_track"]
