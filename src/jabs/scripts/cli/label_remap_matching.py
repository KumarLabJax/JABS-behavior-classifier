"""Match an identity in one pose file to an identity in another over a frame interval.

Used by ``jabs-cli update-pose`` and ``jabs-cli update-labels`` to decide which
destination identity a block of labels belongs to. Two scores are computed for each
candidate destination identity:

- **mask IoU**: per-frame intersection over union of the two identities' filled
  segmentation contours (pose v6+ with segmentation)
- **bbox IoU**: per-frame intersection over union of the two identities' bounding
  boxes (pose v8+)

Each score is the median over the frames in the interval where both identities have
that data. Which score decides the match is controlled by `MatchMetric`.
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
    """

    min_bbox_iou: float = 0.5
    min_mask_iou: float = 0.5
    match_by: MatchMetric = MatchMetric.AUTO


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
