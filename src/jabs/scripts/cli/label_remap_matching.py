"""Match an identity in one pose file to an identity in another over a frame interval.

Used by ``jabs-cli update-pose`` and ``jabs-cli update-labels`` to decide which
destination identity a block of labels belongs to. Two scores are computed for each
candidate destination identity:

- **mask IoU**: per-frame intersection over union of the two identities' filled
  segmentation contours (pose v6+ with segmentation)
- **bbox IoU**: per-frame intersection over union of the two identities' bounding
  boxes (pose v8+)

Which score decides the match is controlled by `MatchMetric`.

By default (`MatchSettings.split_blocks`), the destination identity is chosen frame by
frame and the interval is split wherever the best match changes, because a destination
pose file can hand one animal from one identity to another partway through a labeled
block (for example after a segmentation split). Each run is then accepted or rejected
on its own. With splitting off, the whole interval goes to one identity, scored by the
median over the frames where both identities have data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

logger = logging.getLogger(__name__)

# a frame's segmentation: one (n_vertices, 2) int32 array of (x, y) per contour
Contours = list[npt.NDArray[np.int32]]


class MatchMetric(str, Enum):
    """Which score decides an identity match."""

    AUTO = "auto"
    """Segmentation when both identities have masks on any frame of the interval, else bboxes."""

    SEGMENTATION = "segmentation"
    """Segmentation mask IoU only; intervals without masks do not match."""

    BBOX = "bbox"
    """Bounding box IoU only (the original behavior)."""


@dataclass(frozen=True)
class MatchSettings:
    """Thresholds and metric selection for identity matching.

    Attributes:
        min_bbox_iou: Minimum median bounding box IoU to accept a bbox-decided match.
        min_mask_iou: Minimum median mask IoU to accept a segmentation-decided match.
        match_by: Which score decides the match.
        split_blocks: Split an interval where its best-matching destination identity
            changes, instead of assigning the whole interval to one identity.
        min_run_frames: When splitting, runs shorter than this are folded into a
            neighboring run so brief flickers in the per-frame match do not fragment
            an interval.
    """

    min_bbox_iou: float = 0.5
    min_mask_iou: float = 0.5
    match_by: MatchMetric = MatchMetric.AUTO
    split_blocks: bool = True
    min_run_frames: int = 3


@dataclass(frozen=True)
class RunMatch:
    """A contiguous part of an interval assigned to a single destination identity.

    Attributes:
        start: First frame of the run.
        end: Last frame of the run, inclusive.
        match: The identity match for the run.
    """

    start: int
    end: int
    match: IdentityMatch


@dataclass(frozen=True)
class IdentityMatch:
    """Result of matching one source identity over one interval.

    Attributes:
        identity: Chosen destination identity, or None if nothing could be scored.
        metric: The score that decided the match (`MatchMetric.SEGMENTATION` or
            `MatchMetric.BBOX`).
        score: Median IoU of the chosen identity under the deciding metric.
        threshold: Minimum score required to accept the match.
        bbox_identity: Best destination identity by bounding box IoU, if any bbox frames.
        bbox_iou: Median bbox IoU of ``bbox_identity``.
        mask_identity: Best destination identity by mask IoU, if any mask frames.
        mask_iou: Median mask IoU of ``mask_identity``.
    """

    identity: int | None
    metric: MatchMetric
    score: float
    threshold: float
    bbox_identity: int | None = None
    bbox_iou: float | None = None
    mask_identity: int | None = None
    mask_iou: float | None = None

    @property
    def accepted(self) -> bool:
        """Whether the match meets its threshold."""
        return (
            self.identity is not None and np.isfinite(self.score) and self.score >= self.threshold
        )

    @property
    def disagreement(self) -> bool:
        """Whether segmentation and bounding boxes each pick a different best identity."""
        return (
            self.mask_identity is not None
            and self.bbox_identity is not None
            and self.mask_identity != self.bbox_identity
        )

    def describe(self) -> str:
        """Summarize both scores for log messages, e.g. ``mask IoU 0.82 -> 1, bbox IoU 0.71 -> 1``."""
        parts = []
        if self.mask_identity is not None:
            parts.append(f"mask IoU {self.mask_iou:.2f} -> {self.mask_identity}")
        if self.bbox_identity is not None:
            parts.append(f"bbox IoU {self.bbox_iou:.2f} -> {self.bbox_identity}")
        return ", ".join(parts) if parts else "no overlapping frames"


def bbox_iou(box_a: npt.NDArray[np.floating], box_b: npt.NDArray[np.floating]) -> float:
    """Compute IoU for two boxes shaped (4,) = [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = x2 - x1
    inter_h = y2 - y1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return inter / (area_a + area_b - inter)


def mask_iou(contours_a: Contours, contours_b: Contours) -> float:
    """Compute the IoU of two filled segmentation masks.

    Each mask is the union of its contours filled in, as the segmentation features
    render them. The masks are rasterized only over the bounding box that covers both.

    Args:
        contours_a: Contours of the first mask, each an (n, 2) int32 array of (x, y).
        contours_b: Contours of the second mask.

    Returns:
        Intersection over union of the two masks.
    """
    points_a = np.concatenate(contours_a)
    points_b = np.concatenate(contours_b)
    min_a, max_a = points_a.min(axis=0), points_a.max(axis=0)
    min_b, max_b = points_b.min(axis=0), points_b.max(axis=0)
    # masks whose bounding boxes do not touch cannot overlap
    if np.any(max_a < min_b) or np.any(max_b < min_a):
        return 0.0

    low = np.minimum(min_a, min_b)
    width, height = np.maximum(max_a, max_b) - low + 1
    offset = (-int(low[0]), -int(low[1]))
    mask_a = np.zeros((int(height), int(width)), dtype=np.uint8)
    mask_b = np.zeros_like(mask_a)
    cv2.drawContours(mask_a, contours_a, -1, 1, -1, offset=offset)
    cv2.drawContours(mask_b, contours_b, -1, 1, -1, offset=offset)
    union = np.count_nonzero(mask_a | mask_b)
    return np.count_nonzero(mask_a & mask_b) / union if union else 0.0


def _valid_boxes(boxes: npt.NDArray[np.floating]) -> npt.NDArray[np.bool_]:
    """Return which [x1, y1, x2, y2] boxes are finite with positive width and height."""
    finite = np.isfinite(boxes).all(axis=1)
    positive = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    return finite & positive


class _PoseShapes:
    """Lazily loaded bounding boxes and segmentation contours for one pose file."""

    def __init__(self, pose: PoseEstimation) -> None:
        self._pose = pose
        self._has_bboxes = bool(getattr(pose, "has_bounding_boxes", False)) and hasattr(
            pose, "get_bounding_boxes"
        )
        self._has_segmentation = bool(getattr(pose, "has_segmentation", False)) and hasattr(
            pose, "get_segmentation_data"
        )
        self._bboxes: dict[int, npt.NDArray[np.floating] | None] = {}
        self._seg_data: dict[int, npt.NDArray[np.integer] | None] = {}
        self._contours: dict[tuple[int, int], Contours | None] = {}

    def bboxes(self, identity: int) -> npt.NDArray[np.floating] | None:
        """Per-frame boxes as [x1, y1, x2, y2] for ``identity``, or None if unavailable."""
        if not self._has_bboxes:
            return None
        if identity not in self._bboxes:
            boxes = self._pose.get_bounding_boxes(identity)
            self._bboxes[identity] = None if boxes is None else boxes.reshape(boxes.shape[0], 4)
        return self._bboxes[identity]

    def contours(self, identity: int, frame: int) -> Contours | None:
        """Segmentation contours of ``identity`` on ``frame``, or None if it has none."""
        if not self._has_segmentation:
            return None
        key = (identity, frame)
        if key not in self._contours:
            if identity not in self._seg_data:
                self._seg_data[identity] = self._pose.get_segmentation_data(identity)
            seg = self._seg_data[identity]
            contours: Contours = []
            if seg is not None and frame < len(seg):
                for contour in seg[frame]:
                    valid = np.all(contour != -1, axis=1)
                    if valid.any():
                        contours.append(contour[valid].astype(np.int32))
            self._contours[key] = contours or None
        return self._contours[key]


@dataclass
class _FrameScores:
    """Per-frame scores of every destination identity over one interval.

    Attributes:
        mask: Mask IoU, shape (frames, destination identities); NaN where the source
            has no mask, 0 where only the destination identity has none.
        bbox: Bounding box IoU, with the same conventions as ``mask``.
        mask_frames: Frames decided by masks (the source and some destination have one).
        bbox_frames: Frames where the source and some destination have a box.
        empty_frames: Frames where the source has a mask or box but no destination
            identity can be compared with it; every destination scores 0 on these.
        best: Best destination identity on each frame, or None where nothing overlaps.
    """

    mask: npt.NDArray[np.float64]
    bbox: npt.NDArray[np.float64]
    mask_frames: npt.NDArray[np.bool_]
    bbox_frames: npt.NDArray[np.bool_]
    empty_frames: npt.NDArray[np.bool_]
    best: list[int | None]


def _runs(values: list[int | None]) -> list[tuple[int | None, int]]:
    """Run-length encode ``values`` as ``(value, length)`` pairs."""
    runs: list[tuple[int | None, int]] = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def _merge_adjacent(runs: list[tuple[int | None, int]]) -> list[tuple[int | None, int]]:
    """Combine neighboring runs that have the same value."""
    merged: list[tuple[int | None, int]] = []
    for value, length in runs:
        if merged and merged[-1][0] == value:
            merged[-1] = (value, merged[-1][1] + length)
        else:
            merged.append((value, length))
    return merged


def _fold_short_runs(
    runs: list[tuple[int | None, int]], min_length: int
) -> list[tuple[int | None, int]]:
    """Fold runs without a match, and runs shorter than ``min_length``, into a neighbor.

    The shortest such run is folded first, into its longer neighbor (the earlier one on
    a tie), until every remaining run has a match and is at least ``min_length`` long,
    or only one run is left. Frames keep their order, so the runs still cover the
    whole interval.

    Args:
        runs: ``(identity, length)`` pairs in frame order; identity None means no
            destination identity overlapped those frames.
        min_length: Minimum length of a run that is kept on its own.

    Returns:
        The folded runs.
    """
    runs = _merge_adjacent(runs)
    while len(runs) > 1:
        candidates = [
            (length if value is not None else -1, i)
            for i, (value, length) in enumerate(runs)
            if value is None or length < min_length
        ]
        if not candidates:
            break
        _, i = min(candidates)
        before = runs[i - 1] if i > 0 else None
        after = runs[i + 1] if i + 1 < len(runs) else None
        # fold into a neighbor that has a match, preferring the longer one
        options = [(n, k) for k, n in ((i - 1, before), (i + 1, after)) if n is not None]
        with_match = [(n, k) for n, k in options if n[0] is not None] or options
        target_run, target = max(with_match, key=lambda nk: (nk[0][1], -nk[1]))
        runs[target] = (target_run[0], target_run[1] + runs[i][1])
        del runs[i]
        runs = _merge_adjacent(runs)
    return runs


class IdentityMatcher:
    """Finds the destination identity that best matches a source identity over an interval.

    Args:
        source_pose: Pose file the labels were made against.
        dest_pose: Pose file the labels are being moved to.
        settings: Thresholds and metric selection.
    """

    def __init__(
        self,
        source_pose: PoseEstimation,
        dest_pose: PoseEstimation,
        settings: MatchSettings,
    ) -> None:
        self._source = _PoseShapes(source_pose)
        self._dest = _PoseShapes(dest_pose)
        self._dest_identities = list(dest_pose.identities)
        self._settings = settings

    def _bbox_score(
        self, src_identity: int, dst_identity: int, start: int, end: int
    ) -> float | None:
        """Median bbox IoU over the interval, or None if no frame has both boxes."""
        src_boxes = self._source.bboxes(src_identity)
        dst_boxes = self._dest.bboxes(dst_identity)
        if src_boxes is None or dst_boxes is None:
            return None
        src_slice = src_boxes[start : end + 1]
        dst_slice = dst_boxes[start : end + 1]
        valid = _valid_boxes(src_slice) & _valid_boxes(dst_slice)
        if not np.any(valid):
            return None
        ious = [bbox_iou(a, b) for a, b in zip(src_slice[valid], dst_slice[valid], strict=True)]
        return float(np.median(ious))

    def _mask_score(
        self, src_identity: int, dst_identity: int, start: int, end: int
    ) -> float | None:
        """Median mask IoU over the interval, or None if no frame has both masks."""
        ious = []
        for frame in range(start, end + 1):
            src_contours = self._source.contours(src_identity, frame)
            if src_contours is None:
                continue
            dst_contours = self._dest.contours(dst_identity, frame)
            if dst_contours is not None:
                ious.append(mask_iou(src_contours, dst_contours))
        return float(np.median(ious)) if ious else None

    @staticmethod
    def _best(scores: dict[int, float | None]) -> tuple[int | None, float | None]:
        """Return the first destination identity with the highest score, ignoring None."""
        best_id, best_score = None, None
        for dst_id, score in scores.items():
            if score is not None and (best_score is None or score > best_score):
                best_id, best_score = dst_id, score
        return best_id, best_score

    def match_runs(self, src_identity: int, start: int, end: int) -> list[RunMatch]:
        """Match ``src_identity`` over frames ``start``-``end``, split where the match changes.

        With `MatchSettings.split_blocks` off, this is a single run from `match`.

        Args:
            src_identity: Identity index in the source pose file.
            start: First frame of the interval.
            end: Last frame of the interval, inclusive.

        Returns:
            Runs covering every frame of the interval, in order.
        """
        if not self._settings.split_blocks or not self._dest_identities:
            # with no destination identities, match() returns an unmatched result
            return [RunMatch(start, end, self.match(src_identity, start, end))]

        scores = self._frame_scores(src_identity, start, end)
        runs = _fold_short_runs(_runs(scores.best), self._settings.min_run_frames)
        result = []
        offset = 0
        for identity, length in runs:
            frames = slice(offset, offset + length)
            result.append(
                RunMatch(
                    start + offset,
                    start + offset + length - 1,
                    self._run_match(scores, frames, identity),
                )
            )
            offset += length
        return result

    def _frame_scores(self, src_identity: int, start: int, end: int) -> _FrameScores:
        """Score every destination identity on every frame of the interval.

        A destination identity without a mask (or box) on a frame where the source has
        one scores 0 on that frame. A frame is decided by masks when the source and at
        least one destination identity have masks on it, otherwise by boxes when the
        source and at least one destination identity have boxes.
        """
        match_by = self._settings.match_by
        n_frames = end - start + 1
        n_dest = len(self._dest_identities)
        mask = np.full((n_frames, n_dest), np.nan)
        bbox = np.full((n_frames, n_dest), np.nan)
        mask_frames = np.zeros(n_frames, dtype=bool)
        bbox_frames = np.zeros(n_frames, dtype=bool)
        source_frames = np.zeros(n_frames, dtype=bool)

        if match_by is not MatchMetric.BBOX:
            for i, frame in enumerate(range(start, end + 1)):
                src_contours = self._source.contours(src_identity, frame)
                if src_contours is None:
                    continue
                source_frames[i] = True
                for j, dst in enumerate(self._dest_identities):
                    dst_contours = self._dest.contours(dst, frame)
                    if dst_contours is not None:
                        mask_frames[i] = True
                        mask[i, j] = mask_iou(src_contours, dst_contours)
                    else:
                        mask[i, j] = 0.0

        if match_by is not MatchMetric.SEGMENTATION:
            src_boxes = self._source.bboxes(src_identity)
            if src_boxes is not None:
                src_slice = src_boxes[start : end + 1]
                src_valid = _valid_boxes(src_slice)
                source_frames |= src_valid
                for j, dst in enumerate(self._dest_identities):
                    dst_boxes = self._dest.bboxes(dst)
                    dst_valid = (
                        _valid_boxes(dst_boxes[start : end + 1])
                        if dst_boxes is not None
                        else np.zeros(n_frames, dtype=bool)
                    )
                    for i in np.flatnonzero(src_valid):
                        if dst_valid[i]:
                            bbox_frames[i] = True
                            bbox[i, j] = bbox_iou(src_slice[i], dst_boxes[start + i])
                        else:
                            bbox[i, j] = 0.0

        best: list[int | None] = []
        for i in range(n_frames):
            row = mask[i] if mask_frames[i] else bbox[i] if bbox_frames[i] else None
            if row is None or not np.nanmax(row) > 0:
                best.append(None)
            else:
                best.append(self._dest_identities[int(np.nanargmax(row))])
        empty_frames = source_frames & ~mask_frames & ~bbox_frames
        return _FrameScores(mask, bbox, mask_frames, bbox_frames, empty_frames, best)

    def _run_match(
        self, scores: _FrameScores, frames: slice, identity: int | None
    ) -> IdentityMatch:
        """Score ``identity`` over one run, counting frames where it is missing as 0.

        That includes frames where the source has data but no destination identity
        does, such as frames folded into this run from a run with no match.
        """
        settings = self._settings
        mask_rows = np.flatnonzero(scores.mask_frames[frames]) + frames.start
        bbox_rows = np.flatnonzero(scores.bbox_frames[frames]) + frames.start
        n_empty = int(np.count_nonzero(scores.empty_frames[frames]))

        def with_empty(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Append a 0 per empty frame along the frame axis."""
            zeros = np.zeros((n_empty, *values.shape[1:]))
            return np.concatenate([values, zeros], axis=0)

        def best_of(matrix: npt.NDArray[np.float64], rows: npt.NDArray[np.intp]):
            if len(rows) == 0:
                return None, None
            medians = np.median(with_empty(matrix[rows]), axis=0)
            j = int(np.argmax(medians))
            return self._dest_identities[j], float(medians[j])

        mask_identity, mask_score = best_of(scores.mask, mask_rows)
        bbox_identity, bbox_score = best_of(scores.bbox, bbox_rows)

        # frames decided by boxes because masks could not be compared there
        bbox_only_rows = (
            np.flatnonzero(scores.bbox_frames[frames] & ~scores.mask_frames[frames]) + frames.start
        )
        if len(mask_rows):
            # a run can mix mask frames and box-only frames; score every frame by the
            # score that decided it, so box-only frames are not ignored. The mask
            # threshold applies to the whole run.
            metric, threshold = MatchMetric.SEGMENTATION, settings.min_mask_iou
        else:
            metric = (
                MatchMetric.SEGMENTATION
                if settings.match_by is MatchMetric.SEGMENTATION
                else MatchMetric.BBOX
            )
            threshold = (
                settings.min_mask_iou
                if metric is MatchMetric.SEGMENTATION
                else settings.min_bbox_iou
            )

        if identity is None or len(mask_rows) + len(bbox_only_rows) == 0:
            score = -np.inf
        else:
            j = self._dest_identities.index(identity)
            column = np.concatenate([scores.mask[mask_rows, j], scores.bbox[bbox_only_rows, j]])
            score = float(np.median(with_empty(column)))
        return IdentityMatch(
            identity=identity,
            metric=metric,
            score=score,
            threshold=threshold,
            bbox_identity=bbox_identity,
            bbox_iou=bbox_score,
            mask_identity=mask_identity,
            mask_iou=mask_score,
        )

    def match(self, src_identity: int, start: int, end: int) -> IdentityMatch:
        """Match ``src_identity`` over frames ``start``-``end`` (inclusive).

        Args:
            src_identity: Identity index in the source pose file.
            start: First frame of the interval.
            end: Last frame of the interval, inclusive.

        Returns:
            The match, with both scores for reporting.
        """
        settings = self._settings
        bbox_scores = {
            dst: self._bbox_score(src_identity, dst, start, end) for dst in self._dest_identities
        }
        bbox_identity, bbox_score = self._best(bbox_scores)

        mask_identity, mask_score = None, None
        if settings.match_by is not MatchMetric.BBOX:
            mask_identity, mask_score = self._best(
                {
                    dst: self._mask_score(src_identity, dst, start, end)
                    for dst in self._dest_identities
                }
            )

        if settings.match_by is MatchMetric.BBOX or (
            settings.match_by is MatchMetric.AUTO and mask_identity is None
        ):
            # original behavior: a destination with no usable boxes scores 0.0, and the
            # first destination wins ties, so one is always chosen if any exist
            legacy = {dst: (0.0 if s is None else s) for dst, s in bbox_scores.items()}
            identity, score = self._best(legacy)
            return IdentityMatch(
                identity=identity,
                metric=MatchMetric.BBOX,
                score=-np.inf if score is None else score,
                threshold=settings.min_bbox_iou,
                bbox_identity=bbox_identity,
                bbox_iou=bbox_score,
                mask_identity=mask_identity,
                mask_iou=mask_score,
            )

        return IdentityMatch(
            identity=mask_identity,
            metric=MatchMetric.SEGMENTATION,
            score=-np.inf if mask_score is None else mask_score,
            threshold=settings.min_mask_iou,
            bbox_identity=bbox_identity,
            bbox_iou=bbox_score,
            mask_identity=mask_identity,
            mask_iou=mask_score,
        )
