"""Types for behavior prediction data."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassifierMetadata:
    """Metadata about the classifier used to generate predictions.

    Attributes:
        classifier_file: Path to the classifier file used.
        classifier_hash: Hash of the classifier file for reproducibility.
        app_version: Version of the application that generated the predictions.
        prediction_date: ISO-formatted date/time when predictions were generated.
    """

    classifier_file: str
    classifier_hash: str
    app_version: str
    prediction_date: str


@dataclass(frozen=True)
class BehaviorPrediction:
    """Predictions for a single behavior across all identities in a video.

    Stores predicted classes and class probabilities produced by a classifier,
    along with metadata linking predictions back to the source pose data and
    classifier. Optionally includes post-processed predictions and an
    identity-to-track mapping.

    Attributes:
        behavior: Name of the behavior.
        predicted_class: Predicted class labels, shape (n_identities, n_frames).
        probabilities: Predicted class probabilities, shape (n_identities, n_frames).
        classifier: Metadata about the classifier that produced these predictions.
        pose_file: Name of the pose file these predictions were generated from.
        pose_hash: Hash of the pose file for validation.
        predicted_class_postprocessed: Post-processed predictions, same shape as
            predicted_class. None if post-processing was not applied.
        identity_to_track: Mapping from identity index to track index per frame,
            shape (n_identities, n_frames). None when not applicable.
        external_identity_mapping: Mapping from JABS identity indices to external
            identity labels. None when not applicable.
        extra: Additional metadata not covered by other fields.
    """

    behavior: str
    predicted_class: np.ndarray
    probabilities: np.ndarray
    classifier: ClassifierMetadata
    pose_file: str
    pose_hash: str
    predicted_class_postprocessed: np.ndarray | None = None
    identity_to_track: np.ndarray | None = None
    external_identity_mapping: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes are consistent."""
        if self.predicted_class.ndim != 2:
            raise ValueError(
                f"predicted_class must be 2D (n_identities, n_frames), "
                f"got shape {self.predicted_class.shape}"
            )
        if self.probabilities.shape != self.predicted_class.shape:
            raise ValueError(
                f"probabilities shape {self.probabilities.shape} does not match "
                f"predicted_class shape {self.predicted_class.shape}"
            )
        if (
            self.predicted_class_postprocessed is not None
            and self.predicted_class_postprocessed.shape != self.predicted_class.shape
        ):
            raise ValueError(
                f"predicted_class_postprocessed shape "
                f"{self.predicted_class_postprocessed.shape} does not match "
                f"predicted_class shape {self.predicted_class.shape}"
            )
        if (
            self.identity_to_track is not None
            and self.identity_to_track.shape != self.predicted_class.shape
        ):
            raise ValueError(
                f"identity_to_track shape {self.identity_to_track.shape} "
                f"does not match predicted_class shape "
                f"{self.predicted_class.shape}"
            )
