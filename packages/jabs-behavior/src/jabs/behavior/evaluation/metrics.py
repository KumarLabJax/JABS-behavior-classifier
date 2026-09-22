"""Frame-level and bout-level agreement metrics for classifier evaluation.

Both metric types hold only counts, so they can be summed across identities and
videos and still yield correct rates - averaging per-video rates would weight a
30-frame video the same as a 30-minute one. Derived rates are exposed as
properties computed from the accumulated counts, and are ``None`` rather than
zero when their denominator is empty, so "no data" stays distinguishable from
"nothing was correct".

Frames the classifier could not score (no pose for that identity) and frames the
ground truth leaves unlabeled are excluded from the comparison and reported as
their own counts, rather than silently inflating either agreement or error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..events import ClassLabels
from .bouts import Bout, BoutMatchResult


def _ratio(numerator: int, denominator: int) -> float | None:
    """Divide, returning None for an empty denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.

    Returns:
        The quotient, or None when ``denominator`` is zero.
    """
    return numerator / denominator if denominator else None


def _harmonic_mean(a: float | None, b: float | None) -> float | None:
    """Harmonic mean of two rates, as used for an F1 score.

    Args:
        a: First rate, or None if undefined.
        b: Second rate, or None if undefined.

    Returns:
        The harmonic mean, or None when either input is None. Two rates that are
        both genuinely zero give ``0.0``, not None: "scored zero" has to stay
        distinguishable from "could not be evaluated", which is what None means
        everywhere else here.
    """
    if a is None or b is None:
        return None
    if (a + b) == 0:
        return 0.0
    return 2 * a * b / (a + b)


@dataclass(frozen=True)
class FrameMetrics:
    """Frame-by-frame agreement between predicted classes and ground truth.

    Attributes:
        true_positive: Frames labeled behavior and predicted behavior.
        false_positive: Frames labeled not-behavior but predicted behavior.
        true_negative: Frames labeled not-behavior and predicted not-behavior.
        false_negative: Frames labeled behavior but predicted not-behavior.
        unlabeled_frames: Frames the ground truth leaves unlabeled. Excluded
            from the comparison.
        unpredicted_frames: Frames that are labeled but that the classifier
            produced no prediction for, normally because the identity has no
            pose there. Excluded from the comparison.
    """

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    unlabeled_frames: int = 0
    unpredicted_frames: int = 0

    @property
    def evaluated_frames(self) -> int:
        """Frames that were both labeled and predicted, so could be compared."""
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def accuracy(self) -> float | None:
        """Fraction of evaluated frames where prediction and label agree."""
        return _ratio(self.true_positive + self.true_negative, self.evaluated_frames)

    @property
    def precision_behavior(self) -> float | None:
        """Fraction of frames predicted behavior that are labeled behavior."""
        return _ratio(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall_behavior(self) -> float | None:
        """Fraction of frames labeled behavior that are predicted behavior."""
        return _ratio(self.true_positive, self.true_positive + self.false_negative)

    @property
    def f1_behavior(self) -> float | None:
        """Harmonic mean of behavior precision and recall."""
        return _harmonic_mean(self.precision_behavior, self.recall_behavior)

    @property
    def precision_not_behavior(self) -> float | None:
        """Fraction of frames predicted not-behavior that are labeled not-behavior."""
        return _ratio(self.true_negative, self.true_negative + self.false_negative)

    @property
    def recall_not_behavior(self) -> float | None:
        """Fraction of frames labeled not-behavior that are predicted not-behavior."""
        return _ratio(self.true_negative, self.true_negative + self.false_positive)

    def __add__(self, other: FrameMetrics) -> FrameMetrics:
        """Sum two sets of counts, for accumulating across identities and videos.

        Args:
            other: Metrics to add to this one.

        Returns:
            A new ``FrameMetrics`` holding the summed counts.
        """
        if not isinstance(other, FrameMetrics):
            return NotImplemented
        return FrameMetrics(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            true_negative=self.true_negative + other.true_negative,
            false_negative=self.false_negative + other.false_negative,
            unlabeled_frames=self.unlabeled_frames + other.unlabeled_frames,
            unpredicted_frames=self.unpredicted_frames + other.unpredicted_frames,
        )

    def __radd__(self, other: object) -> FrameMetrics:
        """Add from the right, so ``sum()`` works without an explicit start.

        ``sum()`` begins from the int ``0``; treating any falsy left operand as
        the identity lets a bare ``sum(metrics)`` work instead of raising.

        Args:
            other: The accumulated left operand.

        Returns:
            The sum, or this instance when ``other`` is the zero identity.
        """
        if not other:
            return self
        return self.__add__(other)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, int | float | None]:
        """Return counts and derived rates as a JSON-serializable mapping."""
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "evaluated_frames": self.evaluated_frames,
            "unlabeled_frames": self.unlabeled_frames,
            "unpredicted_frames": self.unpredicted_frames,
            "accuracy": self.accuracy,
            "precision_behavior": self.precision_behavior,
            "recall_behavior": self.recall_behavior,
            "f1_behavior": self.f1_behavior,
            "precision_not_behavior": self.precision_not_behavior,
            "recall_not_behavior": self.recall_not_behavior,
        }


@dataclass(frozen=True)
class BoutMetrics:
    """Bout-level agreement between predicted bouts and ground-truth bouts.

    Detection and precision are computed under many-to-many matching, so a
    ground-truth bout the classifier split in two still counts as detected and
    both fragments count as hits. The splitting is reported separately by
    ``fragmented_truth_bouts``, and the opposite error - one prediction spanning
    several true bouts - by ``merged_predicted_bouts``.

    Attributes:
        criterion_label: Which match criterion produced these counts.
        truth_bouts: Ground-truth behavior bouts found in the labels.
        unevaluable_truth_bouts: Ground-truth bouts lying entirely in frames the
            classifier could not score. Excluded from ``detection_rate``.
        detected_truth_bouts: Evaluable ground-truth bouts matched by at least
            one predicted bout.
        fragmented_truth_bouts: Ground-truth bouts matched by two or more
            predicted bouts.
        predicted_bouts: Predicted behavior bouts.
        unevaluable_predicted_bouts: Predicted bouts lying entirely in unlabeled
            frames, where the ground truth cannot say whether they are correct.
            Excluded from ``precision``.
        matched_predicted_bouts: Predicted bouts matching at least one
            ground-truth bout.
        merged_predicted_bouts: Predicted bouts matching two or more
            ground-truth bouts.
    """

    criterion_label: str
    truth_bouts: int = 0
    unevaluable_truth_bouts: int = 0
    detected_truth_bouts: int = 0
    fragmented_truth_bouts: int = 0
    predicted_bouts: int = 0
    unevaluable_predicted_bouts: int = 0
    matched_predicted_bouts: int = 0
    merged_predicted_bouts: int = 0

    @property
    def evaluable_truth_bouts(self) -> int:
        """Ground-truth bouts that overlap at least one scored frame."""
        return self.truth_bouts - self.unevaluable_truth_bouts

    @property
    def evaluable_predicted_bouts(self) -> int:
        """Predicted bouts that overlap at least one labeled frame."""
        return self.predicted_bouts - self.unevaluable_predicted_bouts

    @property
    def missed_truth_bouts(self) -> int:
        """Evaluable ground-truth bouts with no matching prediction."""
        return self.evaluable_truth_bouts - self.detected_truth_bouts

    @property
    def detection_rate(self) -> float | None:
        """Fraction of evaluable ground-truth bouts that were detected.

        This is bout-level recall.
        """
        return _ratio(self.detected_truth_bouts, self.evaluable_truth_bouts)

    @property
    def precision(self) -> float | None:
        """Fraction of evaluable predicted bouts that match a ground-truth bout."""
        return _ratio(self.matched_predicted_bouts, self.evaluable_predicted_bouts)

    @property
    def f1(self) -> float | None:
        """Harmonic mean of bout detection rate and bout precision."""
        return _harmonic_mean(self.precision, self.detection_rate)

    def __add__(self, other: BoutMetrics) -> BoutMetrics:
        """Sum two sets of counts, for accumulating across identities and videos.

        Args:
            other: Metrics to add to this one. Must share this one's criterion.

        Returns:
            A new ``BoutMetrics`` holding the summed counts.

        Raises:
            ValueError: If the two were computed under different criteria, which
                would make the summed counts meaningless.
        """
        if not isinstance(other, BoutMetrics):
            return NotImplemented
        if self.criterion_label != other.criterion_label:
            raise ValueError(
                f"cannot combine bout metrics from different criteria: "
                f"{self.criterion_label!r} and {other.criterion_label!r}"
            )
        return BoutMetrics(
            criterion_label=self.criterion_label,
            truth_bouts=self.truth_bouts + other.truth_bouts,
            unevaluable_truth_bouts=self.unevaluable_truth_bouts + other.unevaluable_truth_bouts,
            detected_truth_bouts=self.detected_truth_bouts + other.detected_truth_bouts,
            fragmented_truth_bouts=self.fragmented_truth_bouts + other.fragmented_truth_bouts,
            predicted_bouts=self.predicted_bouts + other.predicted_bouts,
            unevaluable_predicted_bouts=(
                self.unevaluable_predicted_bouts + other.unevaluable_predicted_bouts
            ),
            matched_predicted_bouts=self.matched_predicted_bouts + other.matched_predicted_bouts,
            merged_predicted_bouts=self.merged_predicted_bouts + other.merged_predicted_bouts,
        )

    def as_dict(self) -> dict[str, str | int | float | None]:
        """Return counts and derived rates as a JSON-serializable mapping."""
        return {
            "criterion": self.criterion_label,
            "truth_bouts": self.truth_bouts,
            "evaluable_truth_bouts": self.evaluable_truth_bouts,
            "unevaluable_truth_bouts": self.unevaluable_truth_bouts,
            "detected_truth_bouts": self.detected_truth_bouts,
            "missed_truth_bouts": self.missed_truth_bouts,
            "fragmented_truth_bouts": self.fragmented_truth_bouts,
            "predicted_bouts": self.predicted_bouts,
            "evaluable_predicted_bouts": self.evaluable_predicted_bouts,
            "unevaluable_predicted_bouts": self.unevaluable_predicted_bouts,
            "matched_predicted_bouts": self.matched_predicted_bouts,
            "merged_predicted_bouts": self.merged_predicted_bouts,
            "detection_rate": self.detection_rate,
            "precision": self.precision,
            "f1": self.f1,
        }


def compute_frame_metrics(
    truth: npt.NDArray[np.integer],
    predicted: npt.NDArray[np.integer],
) -> FrameMetrics:
    """Compare predicted classes against ground-truth labels frame by frame.

    Only frames that are both labeled and predicted are compared. The rest are
    counted as ``unlabeled_frames`` or ``unpredicted_frames``.

    Args:
        truth: Per-frame ground-truth labels as :class:`ClassLabels` values.
        predicted: Per-frame predicted classes as :class:`ClassLabels` values,
            the same length as ``truth``.

    Returns:
        Frame-level counts for this identity.

    Raises:
        ValueError: If the two arrays have different lengths.
    """
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if truth.shape != predicted.shape:
        raise ValueError(
            f"truth and predicted must have the same shape, "
            f"got {truth.shape} and {predicted.shape}"
        )

    labeled = truth != ClassLabels.NONE
    scored = predicted != ClassLabels.NONE
    evaluable = labeled & scored

    truth_behavior = truth == ClassLabels.BEHAVIOR
    predicted_behavior = predicted == ClassLabels.BEHAVIOR

    return FrameMetrics(
        true_positive=int(np.count_nonzero(evaluable & truth_behavior & predicted_behavior)),
        false_positive=int(np.count_nonzero(evaluable & ~truth_behavior & predicted_behavior)),
        true_negative=int(np.count_nonzero(evaluable & ~truth_behavior & ~predicted_behavior)),
        false_negative=int(np.count_nonzero(evaluable & truth_behavior & ~predicted_behavior)),
        unlabeled_frames=int(np.count_nonzero(~labeled)),
        unpredicted_frames=int(np.count_nonzero(labeled & ~scored)),
    )


def bouts_are_evaluable(
    bouts: Sequence[Bout],
    vector: npt.NDArray[np.integer],
) -> list[bool]:
    """Flag which bouts overlap at least one usable frame in the other vector.

    A ground-truth bout falling entirely where the classifier produced no
    prediction cannot be detected, and a predicted bout falling entirely in
    unlabeled frames cannot be confirmed or refuted. Either would otherwise be
    charged against the classifier for a reason that has nothing to do with its
    quality.

    Args:
        bouts: Bouts to check.
        vector: The opposing per-frame vector - the predicted classes when
            checking ground-truth bouts, the ground-truth labels when checking
            predicted bouts. ``ClassLabels.NONE`` marks the unusable frames.

    Returns:
        One flag per bout, in the order given: True if the bout overlaps at
        least one frame that is not ``ClassLabels.NONE``.
    """
    vector = np.asarray(vector)
    usable = vector != ClassLabels.NONE
    return [bool(np.any(usable[bout.start : bout.end + 1])) for bout in bouts]


def compute_bout_metrics(
    match_result: BoutMatchResult,
    truth_evaluable: Sequence[bool],
    predicted_evaluable: Sequence[bool],
) -> BoutMetrics:
    """Reduce a bout match result to counts.

    Args:
        match_result: Output of :func:`~jabs.behavior.evaluation.bouts.match_bouts`.
        truth_evaluable: One flag per ground-truth bout, from
            :func:`bouts_are_evaluable`.
        predicted_evaluable: One flag per predicted bout, from
            :func:`bouts_are_evaluable`.

    Returns:
        Bout-level counts for this identity under the result's criterion.

    Raises:
        ValueError: If a flag sequence length does not match its bout list.
    """
    if len(truth_evaluable) != len(match_result.truth_matches):
        raise ValueError(
            f"truth_evaluable has {len(truth_evaluable)} entries but there are "
            f"{len(match_result.truth_matches)} ground-truth bouts"
        )
    if len(predicted_evaluable) != len(match_result.predicted_matches):
        raise ValueError(
            f"predicted_evaluable has {len(predicted_evaluable)} entries but there are "
            f"{len(match_result.predicted_matches)} predicted bouts"
        )

    return BoutMetrics(
        criterion_label=match_result.criterion_label,
        truth_bouts=len(match_result.truth_matches),
        unevaluable_truth_bouts=sum(1 for ok in truth_evaluable if not ok),
        detected_truth_bouts=sum(
            1
            for matches, ok in zip(match_result.truth_matches, truth_evaluable, strict=True)
            if ok and matches
        ),
        fragmented_truth_bouts=sum(
            1 for matches in match_result.truth_matches if len(matches) > 1
        ),
        predicted_bouts=len(match_result.predicted_matches),
        unevaluable_predicted_bouts=sum(1 for ok in predicted_evaluable if not ok),
        matched_predicted_bouts=sum(
            1
            for matches, ok in zip(
                match_result.predicted_matches, predicted_evaluable, strict=True
            )
            if ok and matches
        ),
        merged_predicted_bouts=sum(
            1 for matches in match_result.predicted_matches if len(matches) > 1
        ),
    )
