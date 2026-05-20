"""Shared utilities for behavior classifier training and evaluation.

These free functions are used by both binary and multi-class classifiers.
"""

import random
import re
from collections.abc import Generator

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score as _accuracy_score,
)
from sklearn.metrics import (
    confusion_matrix as _confusion_matrix,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import ClassifierType
from jabs.project import TrackLabels

LABEL_THRESHOLD: int = 20


def clean_features(features: pd.DataFrame, classifier_type: ClassifierType) -> pd.DataFrame:
    """Clean features for prediction, handling missing and infinite values.

    Args:
        features: DataFrame of feature data to clean.
        classifier_type: Type of classifier being used.

    Returns:
        Cleaned DataFrame with missing and infinite values handled.
    """
    if classifier_type in (ClassifierType.XGBOOST, ClassifierType.CATBOOST):
        # these classifiers can handle NaN, just replace infinities
        return features.replace([np.inf, -np.inf], np.nan)
    else:
        # Random forests can't handle NAs & infs, so fill them with 0s
        return features.replace([np.inf, -np.inf], 0).fillna(0)


def combine_data(per_frame: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """Combine per-frame and window feature DataFrames.

    Args:
        per_frame: Per-frame features DataFrame.
        window: Window features DataFrame.

    Returns:
        Merged DataFrame with all features concatenated column-wise.
    """
    return pd.concat([per_frame, window], axis=1)


def augment_symmetric(
    features: pd.DataFrame,
    labels: npt.NDArray,
    random_str: str = "ASygRQDZJD",
) -> tuple[pd.DataFrame, npt.NDArray]:
    """Augment features to include L-R and R-L reflection duplicates.

    Features with 'left' or 'right' in their name will have those terms
    swapped to produce mirror-image augmented samples. Features without
    these terms are duplicated unchanged.

    Args:
        features: Feature DataFrame to augment.
        labels: Label array corresponding to features.
        random_str: Temporary replacement string used when swapping left/right.

    Returns:
        Tuple of (augmented features DataFrame, augmented label array),
        each twice the length of the inputs.
    """
    lowercase_features = np.array([x.lower() for x in features.columns.to_list()])
    reflected_feature_names = [re.sub(r"left", random_str, x) for x in lowercase_features]
    reflected_feature_names = [re.sub(r"right", "left", x) for x in reflected_feature_names]
    reflected_feature_names = [re.sub(random_str, "right", x) for x in reflected_feature_names]
    reflected_idxs = [
        np.where(lowercase_features == x)[0][0] if x in lowercase_features else i
        for i, x in enumerate(reflected_feature_names)
    ]
    features_duplicate = features.copy()
    features_duplicate.columns = features.columns.to_numpy()[np.asarray(reflected_idxs)]
    features = pd.concat([features, features_duplicate])
    labels = np.concatenate([labels, labels])
    return features, labels


def downsample_balance(
    features: pd.DataFrame,
    labels: npt.NDArray,
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, npt.NDArray]:
    """Downsample features and labels so that all classes are equally represented.

    Args:
        features: Feature DataFrame to downsample.
        labels: Label array to downsample.
        random_seed: Optional random seed for reproducibility.

    Returns:
        Tuple of (downsampled features DataFrame, downsampled label array).
    """
    label_states, label_counts = np.unique(labels, return_counts=True)
    max_examples_per_class = np.min(label_counts)
    rng = np.random.default_rng(random_seed)
    selected_samples = []
    for cur_label in label_states:
        idxs = np.where(labels == cur_label)[0]
        sampled_idxs = rng.choice(idxs, max_examples_per_class, replace=False)
        selected_samples.append(sampled_idxs)
    selected_samples = np.sort(np.concatenate(selected_samples))
    features = features.iloc[selected_samples]
    labels = labels[selected_samples]
    return features, labels


def leave_one_group_out(
    per_frame_features: pd.DataFrame,
    window_features: pd.DataFrame,
    labels: npt.NDArray,
    groups: npt.NDArray,
    label_threshold: int = LABEL_THRESHOLD,
    min_test_classes: int | None = None,
) -> Generator[dict, None, None]:
    """Implement the leave-one-group-out data splitting strategy.

    A split is accepted only when **both** the test and training portions satisfy
    their respective class-count requirements.

    When ``min_test_classes`` is ``None`` (default, binary mode), a split is
    accepted when:

    - Every class present in the full ``labels`` array appears at least
      ``label_threshold`` times in the test split.
    - The training split also contains every class at least ``label_threshold``
      times (guards against the held-out group being the sole source of a rare
      class).

    When ``min_test_classes`` is an integer (multi-class mode), a split is
    accepted when:

    - The test split contains at least ``min_test_classes`` distinct classes each
      with at least ``label_threshold`` samples.
    - The training split contains **all** classes at least ``label_threshold``
      times (so the model can learn every class regardless of what appears in
      the test split).

    Args:
        per_frame_features: Per-frame feature DataFrame for labeled data.
        window_features: Window feature DataFrame for labeled data.
        labels: Label array corresponding to each feature row.
        groups: Group ID array corresponding to each feature row.
        label_threshold: Minimum number of samples per class required.
        min_test_classes: Minimum number of distinct classes required in the
            test split. ``None`` requires all classes (binary default).

    Yields:
        Dictionary with keys: training_data, training_labels, test_data,
        test_labels, test_group, feature_names.

    Raises:
        ValueError: If no valid split satisfying the threshold can be found.
        ValueError: If ``min_test_classes`` is not ``None`` and is less than 1.
    """
    if min_test_classes is not None and min_test_classes < 1:
        raise ValueError(f"min_test_classes must be None or >= 1, got {min_test_classes}")

    logo = LeaveOneGroupOut()
    x = combine_data(per_frame_features, window_features)
    splits = list(logo.split(x, labels, groups))
    all_classes = np.unique(labels)
    random.shuffle(splits)
    count = 0
    for split in splits:
        test_labels = labels[split[1]]
        if min_test_classes is None:
            # Binary mode: all classes must appear above threshold in both splits.
            test_ok = all(
                np.count_nonzero(test_labels == cls) >= label_threshold for cls in all_classes
            )
            if test_ok:
                # Also require all classes above threshold in the training split
                # so the model can learn every class regardless of the test group.
                train_labels = labels[split[0]]
                test_ok = all(
                    np.count_nonzero(train_labels == cls) >= label_threshold for cls in all_classes
                )
        else:
            # Multi-class mode: test split needs at least min_test_classes
            # classes above threshold; training split must have all classes.
            n_test_classes = sum(
                np.count_nonzero(test_labels == cls) >= label_threshold for cls in all_classes
            )
            train_labels = labels[split[0]]
            train_has_all = all(
                np.count_nonzero(train_labels == cls) >= label_threshold for cls in all_classes
            )
            test_ok = n_test_classes >= min_test_classes and train_has_all

        if test_ok:
            count += 1
            yield {
                "training_data": x.iloc[split[0]],
                "training_labels": labels[split[0]],
                "training_idx": split[0],
                "test_data": x.iloc[split[1]],
                "test_labels": labels[split[1]],
                "test_idx": split[1],
                "test_group": groups[split[1]][0],
                "feature_names": x.columns.to_list(),
            }
    if count == 0:
        raise ValueError("unable to split data")


def accuracy_score(truth: npt.NDArray, predictions: npt.NDArray) -> float:
    """Compute classification accuracy.

    Args:
        truth: Ground-truth label array.
        predictions: Predicted label array.

    Returns:
        Accuracy as a float in [0, 1].
    """
    return _accuracy_score(truth, predictions)


def precision_recall_score(
    truth: npt.NDArray,
    predictions: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute precision, recall, F-score, and support for each class.

    Args:
        truth: Ground-truth label array.
        predictions: Predicted label array.

    Returns:
        Tuple of (precision, recall, f-score, support) arrays as returned
        by sklearn's precision_recall_fscore_support.
    """
    return precision_recall_fscore_support(truth, predictions)


def confusion_matrix(truth: npt.NDArray, predictions: npt.NDArray) -> npt.NDArray:
    """Compute the confusion matrix.

    Args:
        truth: Ground-truth label array.
        predictions: Predicted label array.

    Returns:
        Confusion matrix as a 2D integer array.
    """
    return _confusion_matrix(truth, predictions)


def merge_labels(
    labels_by_behavior: dict[str, npt.NDArray[np.int8]],
    behavior_names: list[str],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Merge per-behavior label arrays into a single multi-class label array.

    Merging rules:
        - ``TrackLabels.Label.BEHAVIOR`` in the ``MULTICLASS_NONE_BEHAVIOR``
          entry → class 0 (background).
        - ``TrackLabels.Label.BEHAVIOR`` in behavior X's entry → class index
          (1-based, by position in ``behavior_names``).
        - All other frames → excluded (not in the returned mask).

    Args:
        labels_by_behavior: dict mapping behavior name to a label array of
            ``TrackLabels.Label`` integer values, one element per frame.
        behavior_names: Ordered list of N behavior names (must not include
            ``MULTICLASS_NONE_BEHAVIOR``).

    Returns:
        Tuple of ``(multiclass_labels, include_mask)`` where:

        - ``multiclass_labels``: integer array of class indices (0..N) for
          the included frames only, length M <= n_frames.
        - ``include_mask``: boolean array of length n_frames; True where the
          frame is included in training.

    Raises:
        ValueError: If ``labels_by_behavior`` is empty, an entry has an
            invalid shape, or any frame is labeled ``BEHAVIOR`` for more than
            one behavior.
    """
    if not labels_by_behavior:
        raise ValueError("labels_by_behavior must not be empty")

    n_frames = next(iter(labels_by_behavior.values())).shape[0]

    for name, arr in labels_by_behavior.items():
        if arr.ndim != 1:
            raise ValueError(f"Label array for '{name}' must be 1-D, got shape {arr.shape}")
        if arr.shape[0] != n_frames:
            raise ValueError(
                f"Label array for '{name}' has length {arr.shape[0]}, expected {n_frames}"
            )

    all_names = [MULTICLASS_NONE_BEHAVIOR, *behavior_names]
    behavior_mask = np.zeros(n_frames, dtype=np.intp)
    for name in all_names:
        if name in labels_by_behavior:
            behavior_mask += (labels_by_behavior[name] == TrackLabels.Label.BEHAVIOR).astype(
                np.intp
            )
    conflict_frames = np.where(behavior_mask > 1)[0]
    if len(conflict_frames) > 0:
        raise ValueError(
            f"Conflicting BEHAVIOR labels found on {len(conflict_frames)} frame(s): "
            f"{conflict_frames.tolist()}. Each frame may be labeled for at "
            f"most one behavior."
        )

    class_indices = np.full(n_frames, -1, dtype=np.intp)

    if MULTICLASS_NONE_BEHAVIOR in labels_by_behavior:
        none_arr = labels_by_behavior[MULTICLASS_NONE_BEHAVIOR]
        class_indices[none_arr == TrackLabels.Label.BEHAVIOR] = 0

    for i, behavior in enumerate(behavior_names, start=1):
        if behavior in labels_by_behavior:
            beh_arr = labels_by_behavior[behavior]
            class_indices[beh_arr == TrackLabels.Label.BEHAVIOR] = i

    include_mask = class_indices >= 0
    return class_indices[include_mask], include_mask
