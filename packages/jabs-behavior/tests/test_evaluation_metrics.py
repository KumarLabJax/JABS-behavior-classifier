"""Tests for frame-level and bout-level evaluation metrics."""

import numpy as np
import pytest

from jabs.behavior.evaluation import (
    Bout,
    BoutMetrics,
    FrameMetrics,
    IoUCriterion,
    OverlapCriterion,
    bouts_are_evaluable,
    compute_bout_metrics,
    compute_frame_metrics,
    extract_bouts,
    match_bouts,
)
from jabs.behavior.events import ClassLabels

NONE = ClassLabels.NONE

# -----------------------------------------------------------------------------
# compute_frame_metrics
# -----------------------------------------------------------------------------


def test_frame_metrics_perfect_agreement() -> None:
    """Frame metrics perfect agreement."""
    v = np.array([0, 1, 1, 0, 1])
    metrics = compute_frame_metrics(v, v)
    assert metrics.true_positive == 3
    assert metrics.true_negative == 2
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.f1_behavior == pytest.approx(1.0)


def test_frame_metrics_counts_each_cell_of_the_confusion_matrix() -> None:
    """Frame metrics counts each cell of the confusion matrix."""
    truth = np.array([1, 1, 0, 0])
    predicted = np.array([1, 0, 1, 0])
    metrics = compute_frame_metrics(truth, predicted)
    assert (metrics.true_positive, metrics.false_negative) == (1, 1)
    assert (metrics.false_positive, metrics.true_negative) == (1, 1)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision_behavior == pytest.approx(0.5)
    assert metrics.recall_behavior == pytest.approx(0.5)


def test_frame_metrics_excludes_unlabeled_frames() -> None:
    """Unlabeled ground truth is counted, not compared."""
    truth = np.array([1, 1, NONE, NONE])
    predicted = np.array([1, 1, 0, 1])
    metrics = compute_frame_metrics(truth, predicted)
    assert metrics.evaluated_frames == 2
    assert metrics.unlabeled_frames == 2
    assert metrics.true_positive == 2
    assert metrics.false_positive == 0  # the NONE frames do not become errors
    assert metrics.accuracy == pytest.approx(1.0)


def test_frame_metrics_excludes_unscored_frames() -> None:
    """Labeled frames the classifier could not score are counted, not compared."""
    truth = np.array([1, 1, 1, 1])
    predicted = np.array([1, 1, NONE, NONE])
    metrics = compute_frame_metrics(truth, predicted)
    assert metrics.evaluated_frames == 2
    assert metrics.unpredicted_frames == 2
    assert metrics.false_negative == 0  # missing pose is not a miss
    assert metrics.recall_behavior == pytest.approx(1.0)


def test_frame_metrics_undefined_rates_are_none_not_zero() -> None:
    """An empty denominator must stay distinguishable from a zero numerator."""
    metrics = compute_frame_metrics(np.array([NONE, NONE]), np.array([1, 1]))
    assert metrics.evaluated_frames == 0
    assert metrics.accuracy is None
    assert metrics.precision_behavior is None
    assert metrics.recall_behavior is None
    assert metrics.f1_behavior is None


def test_frame_metrics_f1_is_none_when_precision_and_recall_are_both_zero() -> None:
    """Frame metrics f1 is none when precision and recall are both zero."""
    metrics = compute_frame_metrics(np.array([1, 0]), np.array([0, 1]))
    assert metrics.precision_behavior == 0.0
    assert metrics.recall_behavior == 0.0
    assert metrics.f1_behavior is None


def test_frame_metrics_rejects_mismatched_lengths() -> None:
    """Frame metrics rejects mismatched lengths."""
    with pytest.raises(ValueError, match="same shape"):
        compute_frame_metrics(np.array([1, 0]), np.array([1, 0, 1]))


def test_frame_metrics_add_sums_counts() -> None:
    """Frame metrics add sums counts."""
    a = compute_frame_metrics(np.array([1, 0]), np.array([1, 0]))
    b = compute_frame_metrics(np.array([1, 1]), np.array([0, 0]))
    total = a + b
    assert total.true_positive == 1
    assert total.true_negative == 1
    assert total.false_negative == 2
    assert total.evaluated_frames == 4


def test_frame_metrics_sum_over_an_empty_iterable() -> None:
    """__radd__ lets sum() start from 0, which aggregation relies on."""
    assert sum([], FrameMetrics()).evaluated_frames == 0


def test_aggregated_rate_is_frame_weighted_not_video_averaged() -> None:
    """A long video must outweigh a short one in the pooled rate."""
    small = compute_frame_metrics(np.array([1]), np.array([0]))  # 0% accurate, 1 frame
    large = compute_frame_metrics(np.ones(99, dtype=int), np.ones(99, dtype=int))  # 100%, 99
    pooled = small + large
    assert pooled.accuracy == pytest.approx(0.99)


# -----------------------------------------------------------------------------
# bouts_are_evaluable
# -----------------------------------------------------------------------------


def test_bouts_are_evaluable_flags_bouts_in_unusable_regions() -> None:
    """Bouts are evaluable flags bouts in unusable regions."""
    bouts = [Bout(0, 2), Bout(5, 7)]
    # the classifier scored the first region but not the second
    predicted = np.array([0, 1, 0, 0, 0, NONE, NONE, NONE])
    assert bouts_are_evaluable(bouts, predicted) == [True, False]


def test_bout_is_evaluable_on_partial_coverage() -> None:
    """One usable frame is enough; the bout is not excluded."""
    predicted = np.array([NONE, NONE, 0])
    assert bouts_are_evaluable([Bout(0, 2)], predicted) == [True]


def test_bouts_are_evaluable_on_empty_list() -> None:
    """Bouts are evaluable on empty list."""
    assert bouts_are_evaluable([], np.zeros(5, dtype=int)) == []


# -----------------------------------------------------------------------------
# compute_bout_metrics
# -----------------------------------------------------------------------------


def _bout_metrics(truth, predicted, criterion) -> BoutMetrics:
    """Run the whole bout pipeline over two vectors."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    truth_bouts = extract_bouts(truth)
    predicted_bouts = extract_bouts(predicted)
    return compute_bout_metrics(
        match_bouts(truth_bouts, predicted_bouts, criterion),
        bouts_are_evaluable(truth_bouts, predicted),
        bouts_are_evaluable(predicted_bouts, truth),
    )


def test_bout_metrics_detects_a_bout_with_disagreeing_boundaries() -> None:
    """The point of bout-level comparison: the edges need not line up."""
    truth = [0, 0, 1, 1, 1, 1, 0, 0]
    predicted = [0, 0, 0, 1, 1, 0, 0, 0]
    metrics = _bout_metrics(truth, predicted, OverlapCriterion(1))
    assert metrics.truth_bouts == 1
    assert metrics.detected_truth_bouts == 1
    assert metrics.detection_rate == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)


def test_bout_metrics_counts_a_missed_bout() -> None:
    """Bout metrics counts a missed bout."""
    truth = [1, 1, 0, 0, 0, 0]
    predicted = [0, 0, 0, 0, 0, 0]
    metrics = _bout_metrics(truth, predicted, OverlapCriterion(1))
    assert metrics.detected_truth_bouts == 0
    assert metrics.missed_truth_bouts == 1
    assert metrics.detection_rate == pytest.approx(0.0)
    assert metrics.precision is None  # nothing was predicted


def test_bout_metrics_reports_fragmentation_without_charging_precision() -> None:
    """One true bout split in two: detected, both fragments count, split reported."""
    truth = [1, 1, 1, 1, 1, 1, 1]
    predicted = [1, 1, 0, 0, 1, 1, 1]
    metrics = _bout_metrics(truth, predicted, OverlapCriterion(1))
    assert metrics.detected_truth_bouts == 1
    assert metrics.detection_rate == pytest.approx(1.0)
    assert metrics.predicted_bouts == 2
    assert metrics.matched_predicted_bouts == 2
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.fragmented_truth_bouts == 1
    assert metrics.merged_predicted_bouts == 0


def test_bout_metrics_reports_a_merged_prediction() -> None:
    """One prediction spanning two true bouts: both detected, merge reported."""
    truth = [1, 1, 0, 0, 1, 1]
    predicted = [1, 1, 1, 1, 1, 1]
    metrics = _bout_metrics(truth, predicted, OverlapCriterion(1))
    assert metrics.truth_bouts == 2
    assert metrics.detected_truth_bouts == 2
    assert metrics.merged_predicted_bouts == 1
    assert metrics.fragmented_truth_bouts == 0


def test_bout_metrics_counts_a_spurious_prediction_against_precision() -> None:
    """Bout metrics counts a spurious prediction against precision."""
    truth = [1, 1, 0, 0, 0, 0]
    predicted = [1, 1, 0, 0, 1, 1]
    metrics = _bout_metrics(truth, predicted, OverlapCriterion(1))
    assert metrics.detection_rate == pytest.approx(1.0)
    assert metrics.predicted_bouts == 2
    assert metrics.matched_predicted_bouts == 1
    assert metrics.precision == pytest.approx(0.5)


def test_bout_metrics_excludes_a_truth_bout_the_classifier_could_not_score() -> None:
    """A bout in a pose gap is not a miss; it is excluded and reported."""
    truth = [1, 1, 0, 0, 1, 1]
    predicted = [1, 1, 0, 0, NONE, NONE]
    metrics = _bout_metrics(truth, predicted, OverlapCriterion(1))
    assert metrics.truth_bouts == 2
    assert metrics.unevaluable_truth_bouts == 1
    assert metrics.evaluable_truth_bouts == 1
    assert metrics.detected_truth_bouts == 1
    assert metrics.detection_rate == pytest.approx(1.0)


def test_bout_metrics_excludes_a_predicted_bout_in_unlabeled_frames() -> None:
    """Ground truth cannot refute a prediction where it says nothing."""
    truth = [1, 1, 0, 0, NONE, NONE]
    predicted = [1, 1, 0, 0, 1, 1]
    metrics = _bout_metrics(truth, predicted, OverlapCriterion(1))
    assert metrics.predicted_bouts == 2
    assert metrics.unevaluable_predicted_bouts == 1
    assert metrics.evaluable_predicted_bouts == 1
    assert metrics.matched_predicted_bouts == 1
    assert metrics.precision == pytest.approx(1.0)


def test_iou_criterion_yields_a_lower_detection_rate_than_overlap() -> None:
    """The two criteria disagree exactly where boundary quality is poor."""
    truth = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # 8 frames
    predicted = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  # 2 frames -> IoU 0.25

    overlap = _bout_metrics(truth, predicted, OverlapCriterion(1))
    iou = _bout_metrics(truth, predicted, IoUCriterion(0.5))

    assert overlap.detection_rate == pytest.approx(1.0)
    assert iou.detection_rate == pytest.approx(0.0)


def test_bout_metrics_empty_inputs() -> None:
    """Bout metrics empty inputs."""
    metrics = _bout_metrics([0, 0, 0], [0, 0, 0], OverlapCriterion(1))
    assert metrics.truth_bouts == 0
    assert metrics.predicted_bouts == 0
    assert metrics.detection_rate is None
    assert metrics.precision is None
    assert metrics.f1 is None


def test_bout_metrics_add_sums_counts() -> None:
    """Bout metrics add sums counts."""
    a = _bout_metrics([1, 1, 0, 0], [1, 1, 0, 0], OverlapCriterion(1))
    b = _bout_metrics([1, 1, 0, 0], [0, 0, 0, 0], OverlapCriterion(1))
    total = a + b
    assert total.truth_bouts == 2
    assert total.detected_truth_bouts == 1
    assert total.detection_rate == pytest.approx(0.5)


def test_bout_metrics_refuses_to_combine_different_criteria() -> None:
    """Bout metrics refuses to combine different criteria."""
    a = _bout_metrics([1, 1], [1, 1], OverlapCriterion(1))
    b = _bout_metrics([1, 1], [1, 1], IoUCriterion(0.5))
    with pytest.raises(ValueError, match="different criteria"):
        _ = a + b


def test_compute_bout_metrics_rejects_mismatched_flag_lengths() -> None:
    """Compute bout metrics rejects mismatched flag lengths."""
    result = match_bouts([Bout(0, 2)], [Bout(0, 2)], OverlapCriterion(1))
    with pytest.raises(ValueError, match="truth_evaluable has"):
        compute_bout_metrics(result, [True, True], [True])
    with pytest.raises(ValueError, match="predicted_evaluable has"):
        compute_bout_metrics(result, [True], [])
