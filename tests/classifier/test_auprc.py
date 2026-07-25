"""Tests for AUPRC (average precision) metrics in classifier_utils."""

import numpy as np

from jabs.classifier.classifier_utils import binary_auprc, macro_auprc


def test_binary_auprc_perfect_separation():
    """Perfectly-ranked positives give AUPRC 1.0; behavior is the larger label."""
    classes = np.array([0, 1])
    truth = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])  # col 1 = behavior
    assert binary_auprc(truth, proba, classes) == 1.0


def test_binary_auprc_uses_behavior_column():
    """A ranking that inverts positives/negatives scores below the base rate."""
    classes = np.array([0, 1])
    truth = np.array([0, 0, 1, 1])
    # behavior prob HIGHER for the negatives -> worst-case ranking
    proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.9, 0.1]])
    ap = binary_auprc(truth, proba, classes)
    assert ap < 0.6  # base rate is 0.5; an inverted ranking is worse


def test_binary_auprc_single_class_is_nan():
    """A test fold with only one class present yields NaN (AUPRC undefined)."""
    classes = np.array([0, 1])
    truth = np.array([1, 1, 1])  # no negatives
    proba = np.array([[0.2, 0.8], [0.1, 0.9], [0.4, 0.6]])
    assert np.isnan(binary_auprc(truth, proba, classes))


def test_binary_auprc_positive_is_larger_label_not_zero_one():
    """Behavior is the larger label even when labels are e.g. {1, 2}."""
    classes = np.array([1, 2])  # behavior == 2 (col 1)
    truth = np.array([1, 1, 2, 2])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    assert binary_auprc(truth, proba, classes) == 1.0


def test_macro_auprc_averages_present_classes():
    """Macro AUPRC averages per-class one-vs-rest AP over classes with both labels."""
    classes = np.array([0, 1, 2])
    truth = np.array([0, 1, 2, 0, 1, 2])
    # each class perfectly separable in its own column
    proba = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    assert macro_auprc(truth, proba, classes) == 1.0


def test_macro_auprc_all_single_class_is_nan():
    """If no class has both positive and negative examples, macro AUPRC is NaN."""
    classes = np.array([0, 1])
    truth = np.array([0, 0])
    proba = np.array([[0.9, 0.1], [0.8, 0.2]])
    assert np.isnan(macro_auprc(truth, proba, classes))
