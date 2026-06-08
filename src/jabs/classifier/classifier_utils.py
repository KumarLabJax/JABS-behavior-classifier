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


def logo_split_is_valid(
    test_labels: npt.NDArray,
    train_labels: npt.NDArray,
    all_classes: npt.NDArray,
    label_threshold: int,
    min_test_classes: int | None,
) -> bool:
    """Return True if a leave-one-group-out split satisfies threshold criteria.

    The training portion must always contain every class above
    ``label_threshold`` so the model can learn each class regardless of the
    held-out group.

    The test portion criterion depends on ``min_test_classes``:

    - ``None`` (binary default): every class must be above ``label_threshold``.
    - integer (multi-class): at least ``min_test_classes`` distinct classes
      must be above ``label_threshold``.

    Args:
        test_labels: Label array for the test split.
        train_labels: Label array for the training split.
        all_classes: All class values to consider (typically ``np.unique`` of
            the full label array).
        label_threshold: Minimum number of samples per class required.
        min_test_classes: Minimum number of distinct classes required in the
            test split, or ``None`` to require every class.

    Returns:
        True if the split is acceptable for cross-validation.
    """
    train_has_all = all(
        np.count_nonzero(train_labels == cls) >= label_threshold for cls in all_classes
    )
    if not train_has_all:
        return False
    if min_test_classes is None:
        return all(np.count_nonzero(test_labels == cls) >= label_threshold for cls in all_classes)
    n_test_classes = sum(
        np.count_nonzero(test_labels == cls) >= label_threshold for cls in all_classes
    )
    return n_test_classes >= min_test_classes


def leave_one_group_out(
    per_frame_features: pd.DataFrame,
    window_features: pd.DataFrame,
    labels: npt.NDArray,
    groups: npt.NDArray,
    label_threshold: int = LABEL_THRESHOLD,
    min_test_classes: int | None = None,
    excluded_groups: set[int] | None = None,
) -> Generator[dict, None, None]:
    """Implement the leave-one-group-out data splitting strategy.

    A split is accepted only when both the test and training portions satisfy
    :func:`logo_split_is_valid`.

    Args:
        per_frame_features: Per-frame feature DataFrame for labeled data.
        window_features: Window feature DataFrame for labeled data.
        labels: Label array corresponding to each feature row.
        groups: Group ID array corresponding to each feature row.
        label_threshold: Minimum number of samples per class required.
        min_test_classes: Minimum number of distinct classes required in the
            test split. ``None`` requires all classes (binary default).
        excluded_groups: Group ids whose rows are held out of training. These
            groups may still serve as the test (held-out) group, but their rows
            are never included in any training fold.

    Yields:
        Dictionary with keys: training_data, training_labels, test_data,
        test_labels, test_group, feature_names.

    Raises:
        ValueError: If no valid split satisfying the threshold can be found.
        ValueError: If ``min_test_classes`` is not ``None`` and is less than 1.
    """
    if min_test_classes is not None and min_test_classes < 1:
        raise ValueError(f"min_test_classes must be None or >= 1, got {min_test_classes}")

    excluded = list(excluded_groups) if excluded_groups else []
    logo = LeaveOneGroupOut()
    x = combine_data(per_frame_features, window_features)
    splits = list(logo.split(x, labels, groups))
    all_classes = np.unique(labels)
    random.shuffle(splits)
    count = 0
    for train_idx, test_idx in splits:
        # excluded groups never contribute to a training fold (but the held-out
        # group itself may be an excluded group)
        if excluded:
            train_idx = train_idx[~np.isin(groups[train_idx], excluded)]
        if not logo_split_is_valid(
            labels[test_idx], labels[train_idx], all_classes, label_threshold, min_test_classes
        ):
            continue
        count += 1
        yield {
            "training_data": x.iloc[train_idx],
            "training_labels": labels[train_idx],
            "training_idx": train_idx,
            "test_data": x.iloc[test_idx],
            "test_labels": labels[test_idx],
            "test_idx": test_idx,
            "test_group": groups[test_idx][0],
            "feature_names": x.columns.to_list(),
        }
    if count == 0:
        raise ValueError("unable to split data")


def count_valid_logo_splits(
    labels: npt.NDArray,
    groups: npt.NDArray,
    label_threshold: int = LABEL_THRESHOLD,
    min_test_classes: int | None = None,
    excluded_groups: set[int] | None = None,
) -> int:
    """Count groups that would yield a valid LOGO split.

    Mirrors the per-split acceptance rule used by :func:`leave_one_group_out`
    without constructing feature matrices, so callers can pre-flight how many
    iterations a CV run will produce.

    Args:
        labels: Label array corresponding to each frame.
        groups: Group ID array corresponding to each label.
        label_threshold: Minimum number of samples per class required.
        min_test_classes: Minimum number of distinct classes required in the
            test split, or ``None`` to require every class.
        excluded_groups: Group ids whose rows are held out of training. These
            groups may still serve as the test group, but their rows never count
            toward a training fold.

    Returns:
        Number of groups that can serve as a valid LOGO test split.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    all_classes = np.unique(labels)
    unique_groups = np.unique(groups)
    excluded_mask = (
        np.isin(groups, list(excluded_groups))
        if excluded_groups
        else np.zeros(len(groups), dtype=bool)
    )
    count = 0
    for g in unique_groups:
        test_mask = groups == g
        # the training portion excludes both the held-out group and any
        # excluded-video rows
        train_mask = ~test_mask & ~excluded_mask
        if logo_split_is_valid(
            labels[test_mask],
            labels[train_mask],
            all_classes,
            label_threshold,
            min_test_classes,
        ):
            count += 1
    return count


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
