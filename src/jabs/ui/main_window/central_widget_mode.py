"""Per-mode dispatch helpers for CentralWidget label and prediction operations.

CentralWidget hosts both binary and multi-class workflows. Each helper here
encapsulates one operation that diverges by classifier mode so the widget
itself can call the operation uniformly and the mode check lives in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierMode

from ..behavior_timeline import track_labels_to_lut_indices

if TYPE_CHECKING:
    from jabs.project import VideoLabels
    from jabs.project.prediction_manager import PredictionManager


def load_video_predictions(
    prediction_manager: PredictionManager,
    mode: ClassifierMode,
    video_name: str,
    behavior: str,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[str] | None,
]:
    """Load saved predictions for ``video_name``, dispatching by classifier mode.

    Args:
        prediction_manager: The project's prediction manager.
        mode: Active classifier mode.
        video_name: File name of the video to load predictions for.
        behavior: Behavior key; only used in binary mode.

    Returns:
        ``(predictions, probabilities, postprocessed_predictions, class_names)``.
        ``class_names`` is the ordered class list in multi-class mode, ``None``
        in binary mode.
    """
    if mode == ClassifierMode.MULTICLASS:
        return prediction_manager.load_multiclass_predictions(video_name)
    predictions, probabilities, postprocessed = prediction_manager.load_predictions(
        video_name, behavior
    )
    return predictions, probabilities, postprocessed, None


def apply_behavior_label(
    labels: VideoLabels,
    mode: ClassifierMode,
    identity_str: str,
    current_behavior: str,
    start: int,
    end: int,
) -> None:
    """Apply the current-behavior label across ``[start, end]``.

    In multi-class mode, competing behavior labels on the same range are
    cleared first to maintain mutual exclusivity across classes. Binary mode
    just labels the requested track.
    """
    if mode == ClassifierMode.MULTICLASS:
        for behavior, track in labels.iter_behavior_labels(identity_str):
            if behavior != current_behavior:
                track.clear_labels(start, end)
    labels.get_track_labels(identity_str, current_behavior).label_behavior(start, end)


def apply_not_behavior_label(
    labels: VideoLabels,
    mode: ClassifierMode,
    identity_str: str,
    current_behavior: str,
    start: int,
    end: int,
) -> tuple[str, bool]:
    """Apply the mode-appropriate negative label.

    In binary mode this is a true "not behavior" label on the current behavior
    track. In multi-class mode the action becomes an explicit None label
    (a positive label on the reserved :data:`MULTICLASS_NONE_BEHAVIOR` track);
    competing behavior labels on the same range are cleared first.

    Returns:
        ``(behavior_key, is_positive_label)`` for session-tracker logging.
        Binary mode returns ``(current_behavior, False)``; multi-class mode
        returns ``(MULTICLASS_NONE_BEHAVIOR, True)``.
    """
    if mode == ClassifierMode.MULTICLASS:
        for behavior, track in labels.iter_behavior_labels(identity_str):
            if behavior != MULTICLASS_NONE_BEHAVIOR:
                track.clear_labels(start, end)
        labels.get_track_labels(identity_str, MULTICLASS_NONE_BEHAVIOR).label_behavior(start, end)
        return MULTICLASS_NONE_BEHAVIOR, True
    labels.get_track_labels(identity_str, current_behavior).label_not_behavior(start, end)
    return current_behavior, False


def build_timeline_label_arrays(
    labels: VideoLabels,
    mode: ClassifierMode,
    num_identities: int,
    current_behavior: str,
    behaviors: list[str],
) -> list[npt.NDArray]:
    """Build per-identity label arrays for the timeline widget.

    Binary mode returns one LUT-index array per identity covering
    ``current_behavior`` only. Multi-class mode returns one merged
    multi-class label array per identity covering all ``behaviors``.
    """
    if mode == ClassifierMode.MULTICLASS:
        return [
            labels.build_multiclass_label_array(str(i), behaviors) for i in range(num_identities)
        ]
    return [
        track_labels_to_lut_indices(labels.get_track_labels(str(i), current_behavior))
        for i in range(num_identities)
    ]
