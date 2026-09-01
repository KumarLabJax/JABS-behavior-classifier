"""Full-sequence inference for a single identity.

This is the inference step shared by prediction and by cross-validation's
postprocessing evaluation. Both need predictions over *every* frame of a
video for one identity - not just the labeled frames - because the
postprocessing pipeline reasons about contiguous bouts and about runs of
frames that have no prediction at all.

Keeping it in one place means the metrics cross-validation reports for the
postprocessing pipeline are computed from the same predictions the classify
path would produce for the same identity and model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from jabs.feature_extraction import IdentityFeatures

    from .protocols import ClassifierProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentityPrediction:
    """Per-frame prediction output for one identity over a full video.

    All three arrays are indexed by global frame number and span every frame of
    the video. Frames where the identity does not exist carry a ``-1``
    prediction and zero probability.

    Attributes:
        probabilities: Full per-class probability matrix, shape
            ``(n_frames, n_classes)``.
        predictions: Predicted class index per frame, ``-1`` where the identity
            has no pose.
        confidence: Probability of the chosen class per frame.
    """

    probabilities: npt.NDArray[np.float32]
    predictions: npt.NDArray[np.int8]
    confidence: npt.NDArray[np.float32]


def predict_identity(
    classifier: ClassifierProtocol,
    identity_features: IdentityFeatures,
    window_size: int,
) -> IdentityPrediction | None:
    """Predict every frame of one identity's track.

    Args:
        classifier: Trained classifier used for inference.
        identity_features: Feature accessor for the identity being predicted.
        window_size: Window size to use for window features.

    Returns:
        The per-frame prediction arrays, or ``None`` when the identity has no
        feature rows at all (callers substitute a zero-filled result sized to
        the video's frame count).
    """
    feature_values = identity_features.get_features(window_size)
    per_frame_features = pd.DataFrame(feature_values["per_frame"])
    window_features = pd.DataFrame(feature_values["window"])
    data = classifier.combine_data(per_frame_features, window_features)

    if data.shape[0] == 0:
        return None

    probabilities = classifier.predict_proba(data, feature_values["frame_indexes"])
    predictions, confidence = classifier.derive_predictions(probabilities)
    return IdentityPrediction(
        probabilities=probabilities,
        predictions=predictions,
        confidence=confidence,
    )
