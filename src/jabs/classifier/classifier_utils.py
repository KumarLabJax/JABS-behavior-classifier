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

from jabs.core.enums import ClassifierType

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
    selected_samples = []
    for cur_label in label_states:
        idxs = np.where(labels == cur_label)[0]
        rng = np.random.default_rng(random_seed)
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
) -> Generator[dict, None, None]:
    """Implement the leave-one-group-out data splitting strategy.

    Splits are filtered so that the test set has at least `label_threshold`
    samples for each class present in the test split. This ensures meaningful
    evaluation regardless of the number of behavior classes.

    Args:
        per_frame_features: Per-frame feature DataFrame for labeled data.
        window_features: Window feature DataFrame for labeled data.
        labels: Label array corresponding to each feature row.
        groups: Group ID array corresponding to each feature row.
        label_threshold: Minimum number of samples per class in the test split.

    Yields:
        Dictionary with keys: training_data, training_labels, test_data,
        test_labels, test_group, feature_names.

    Raises:
        ValueError: If no valid split satisfying the threshold can be found.
    """
    logo = LeaveOneGroupOut()
    x = combine_data(per_frame_features, window_features)
    splits = list(logo.split(x, labels, groups))
    random.shuffle(splits)
    count = 0
    for split in splits:
        test_labels = labels[split[1]]
        unique_classes = np.unique(test_labels)
        if all(np.count_nonzero(test_labels == cls) >= label_threshold for cls in unique_classes):
            count += 1
            yield {
                "training_data": x.iloc[split[0]],
                "training_labels": labels[split[0]],
                "test_data": x.iloc[split[1]],
                "test_labels": labels[split[1]],
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
