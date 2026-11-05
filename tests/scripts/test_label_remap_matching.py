"""Tests for identity matching used to remap labels between pose files."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

from jabs.scripts.cli.label_remap_matching import (
    IdentityMatcher,
    MatchMetric,
    MatchSettings,
    bbox_iou,
    mask_iou,
)

_PAD = -1


def _rect(x0: int, y0: int, x1: int, y1: int) -> npt.NDArray[np.int32]:
    """Rectangle contour covering pixels x0..x1, y0..y1 when filled."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


def _seg_array(frames: list[list[npt.NDArray[np.int32]] | None]) -> npt.NDArray[np.int32]:
    """Pack per-frame contour lists into a -1 padded (frames, contours, vertices, 2) array."""
    n_contours = max((len(f) for f in frames if f), default=1)
    n_vertices = max((len(c) for f in frames if f for c in f), default=1)
    seg = np.full((len(frames), n_contours, n_vertices, 2), _PAD, dtype=np.int32)
    for i, contours in enumerate(frames):
        for j, contour in enumerate(contours or []):
            seg[i, j, : len(contour)] = contour
    return seg


def _box(x0: float, y0: float, x1: float, y1: float) -> npt.NDArray[np.float32]:
    """One frame's bounding box in the [[x1, y1], [x2, y2]] layout pose files use."""
    return np.array([[x0, y0], [x1, y1]], dtype=np.float32)


def _pose(
    boxes: dict[int, npt.NDArray[np.float32]] | None = None,
    masks: dict[int, list[list[npt.NDArray[np.int32]] | None]] | None = None,
) -> SimpleNamespace:
    """Fake pose file with optional per-identity boxes (frames, 2, 2) and masks."""
    identities = sorted(set(boxes or {}) | set(masks or {}))
    seg = {i: _seg_array(m) for i, m in (masks or {}).items()}
    return SimpleNamespace(
        identities=identities,
        has_bounding_boxes=boxes is not None,
        get_bounding_boxes=lambda i: boxes[i],
        has_segmentation=masks is not None,
        get_segmentation_data=lambda i: seg.get(i),
    )


# ---------------------------------------------------------------------------
# mask_iou
# ---------------------------------------------------------------------------


def test_mask_iou_identical_masks() -> None:
    """A mask matches itself exactly."""
    assert mask_iou([_rect(0, 0, 9, 9)], [_rect(0, 0, 9, 9)]) == pytest.approx(1.0)


def test_mask_iou_counts_overlapping_pixels() -> None:
    """Two 10x10 squares offset by 5 pixels overlap in 50 of 150 pixels."""
    assert mask_iou([_rect(0, 0, 9, 9)], [_rect(5, 0, 14, 9)]) == pytest.approx(50 / 150)


def test_mask_iou_disjoint_masks() -> None:
    """Masks that do not touch have zero overlap."""
    assert mask_iou([_rect(0, 0, 9, 9)], [_rect(100, 100, 109, 109)]) == 0.0


def test_mask_iou_fills_every_contour_of_a_mask() -> None:
    """A mask split into separate pieces counts all of them."""
    two_pieces = [_rect(0, 0, 9, 9), _rect(20, 0, 29, 9)]
    assert mask_iou(two_pieces, [_rect(0, 0, 9, 9)]) == pytest.approx(0.5)


def test_bbox_iou_half_overlap() -> None:
    """bbox_iou uses continuous coordinates: two 10x10 boxes offset by 5 give 50/150."""
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([5.0, 0.0, 15.0, 10.0])
    assert bbox_iou(a, b) == pytest.approx(50 / 150)


# ---------------------------------------------------------------------------
# IdentityMatcher
# ---------------------------------------------------------------------------


def _diagonal(n: int) -> npt.NDArray[np.int32]:
    """A thin diagonal band from the top-left to the bottom-right of an n x n box."""
    return np.array([[0, 0], [3, 0], [n, n - 3], [n, n], [n - 3, n], [0, 3]], dtype=np.int32)


def _anti_diagonal(n: int) -> npt.NDArray[np.int32]:
    """A thin diagonal band from the top-right to the bottom-left of an n x n box."""
    return np.array([[n, 0], [n, 3], [3, n], [0, n], [0, n - 3], [n - 3, 0]], dtype=np.int32)


@pytest.fixture
def crossed_mice() -> tuple[SimpleNamespace, SimpleNamespace]:
    """Two mice whose bounding boxes favor the wrong destination identity.

    Source identity 0 is a diagonal band in a 40x40 box. In the destination, identity
    0 has the same mask but a slightly offset box, while identity 1 has exactly the
    source's box but lies along the other diagonal, so its mask barely overlaps.
    """
    frames = 5
    src = _pose(
        boxes={0: np.stack([_box(0, 0, 40, 40)] * frames)},
        masks={0: [[_diagonal(40)]] * frames},
    )
    dst = _pose(
        boxes={
            0: np.stack([_box(2, 2, 42, 42)] * frames),
            1: np.stack([_box(0, 0, 40, 40)] * frames),
        },
        masks={0: [[_diagonal(40)]] * frames, 1: [[_anti_diagonal(40)]] * frames},
    )
    return src, dst


def test_auto_prefers_segmentation_and_flags_disagreement(crossed_mice) -> None:
    """With masks available, the mask decides, and disagreeing boxes are reported."""
    src, dst = crossed_mice
    match = IdentityMatcher(src, dst, MatchSettings()).match(0, 0, 4)

    assert match.metric is MatchMetric.SEGMENTATION
    assert match.identity == 0
    assert match.score == pytest.approx(1.0)
    assert match.bbox_identity == 1
    assert match.disagreement
    assert match.accepted


def test_bbox_mode_ignores_segmentation(crossed_mice) -> None:
    """--match-by bbox reproduces the box-only choice."""
    src, dst = crossed_mice
    settings = MatchSettings(match_by=MatchMetric.BBOX)
    match = IdentityMatcher(src, dst, settings).match(0, 0, 4)

    assert match.metric is MatchMetric.BBOX
    assert match.identity == 1
    assert match.mask_identity is None
    assert not match.disagreement


def test_auto_falls_back_to_bbox_without_masks() -> None:
    """Pose files without segmentation are matched by bounding boxes."""
    boxes = {0: np.stack([_box(0, 0, 10, 10)] * 3)}
    match = IdentityMatcher(_pose(boxes=boxes), _pose(boxes=boxes), MatchSettings()).match(0, 0, 2)

    assert match.metric is MatchMetric.BBOX
    assert match.identity == 0
    assert match.score == pytest.approx(1.0)


def test_auto_falls_back_to_bbox_when_block_has_no_mask_frames() -> None:
    """Masks elsewhere in the video do not matter; only the block's frames count."""
    boxes = {0: np.stack([_box(0, 0, 10, 10)] * 4)}
    masks = {0: [None, None, [_rect(0, 0, 9, 9)], [_rect(0, 0, 9, 9)]]}
    matcher = IdentityMatcher(_pose(boxes, masks), _pose(boxes, masks), MatchSettings())

    assert matcher.match(0, 0, 1).metric is MatchMetric.BBOX
    assert matcher.match(0, 2, 3).metric is MatchMetric.SEGMENTATION


def test_segmentation_mode_rejects_blocks_without_masks() -> None:
    """--match-by segmentation does not fall back to boxes."""
    boxes = {0: np.stack([_box(0, 0, 10, 10)] * 2)}
    settings = MatchSettings(match_by=MatchMetric.SEGMENTATION)
    match = IdentityMatcher(_pose(boxes=boxes), _pose(boxes=boxes), settings).match(0, 0, 1)

    assert match.metric is MatchMetric.SEGMENTATION
    assert match.identity is None
    assert not match.accepted


def test_mask_score_uses_only_frames_where_both_have_masks() -> None:
    """Frames missing a mask on either side are left out of the median."""
    src = _pose(masks={0: [[_rect(0, 0, 9, 9)], None, [_rect(0, 0, 9, 9)]]})
    dst = _pose(masks={0: [[_rect(0, 0, 9, 9)], [_rect(50, 50, 59, 59)], None]})

    match = IdentityMatcher(src, dst, MatchSettings()).match(0, 0, 2)

    assert match.identity == 0
    assert match.score == pytest.approx(1.0)


def test_mask_threshold_is_separate_from_bbox_threshold() -> None:
    """A segmentation-decided match is held to --min-mask-iou, not --min-iou-thresh."""
    src = _pose(masks={0: [[_rect(0, 0, 9, 9)]]})
    dst = _pose(masks={0: [[_rect(5, 0, 14, 9)]]})  # mask IoU 1/3

    lenient = MatchSettings(min_bbox_iou=0.9, min_mask_iou=0.3)
    strict = MatchSettings(min_bbox_iou=0.1, min_mask_iou=0.5)

    assert IdentityMatcher(src, dst, lenient).match(0, 0, 0).accepted
    assert not IdentityMatcher(src, dst, strict).match(0, 0, 0).accepted


# ---------------------------------------------------------------------------
# bbox mode reproduces the original algorithm
# ---------------------------------------------------------------------------


def _reference_find_best_identity(src_pose, dst_pose, src_identity, start, end):
    """The original update-pose bbox matcher, kept as a reference."""

    def boxes(pose, identity):
        b = pose.get_bounding_boxes(identity)
        return b.reshape(b.shape[0], 4)

    def valid(b):
        return np.isfinite(b).all(axis=1) & (b[:, 2] - b[:, 0] > 0) & (b[:, 3] - b[:, 1] > 0)

    best_id, best_score = None, -float("inf")
    for dst_id in dst_pose.identities:
        s = boxes(src_pose, src_identity)[start : end + 1]
        d = boxes(dst_pose, dst_id)[start : end + 1]
        mask = valid(s) & valid(d)
        iou = (
            float(np.median([bbox_iou(a, b) for a, b in zip(s[mask], d[mask], strict=True)]))
            if np.any(mask)
            else 0.0
        )
        if iou > best_score:
            best_id, best_score = dst_id, iou
    return best_id, best_score


@pytest.mark.parametrize("match_by", [MatchMetric.BBOX, MatchMetric.AUTO])
def test_bbox_matching_equals_original_algorithm(match_by: MatchMetric) -> None:
    """Without segmentation, matches are identical to the original bbox matcher."""
    rng = np.random.default_rng(seed=4)
    frames = 40

    def random_boxes() -> npt.NDArray[np.float32]:
        corner = rng.uniform(0, 100, (frames, 2))
        size = rng.uniform(-5, 30, (frames, 2))  # some zero/negative-area boxes
        boxes = np.stack([corner, corner + size], axis=1).astype(np.float32)
        boxes[rng.random(frames) < 0.2] = np.nan
        return boxes

    src = _pose(boxes={i: random_boxes() for i in range(3)})
    dst = _pose(boxes={i: random_boxes() for i in range(4)})
    matcher = IdentityMatcher(src, dst, MatchSettings(match_by=match_by))

    for src_identity in range(3):
        for start, end in [(0, 0), (0, 9), (5, 30), (12, 39), (20, 20)]:
            match = matcher.match(src_identity, start, end)
            ref_id, ref_score = _reference_find_best_identity(src, dst, src_identity, start, end)
            assert (match.identity, match.score) == (ref_id, ref_score)


# ---------------------------------------------------------------------------
# splitting blocks where the best match changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("runs", "min_length", "expected"),
    [
        ([(1, 14), (2, 17)], 3, [(1, 14), (2, 17)]),
        ([(0, 10), (1, 1), (0, 10)], 3, [(0, 21)]),
        ([(0, 10), (1, 2), (2, 10)], 3, [(0, 12), (2, 10)]),
        ([(1, 2), (0, 10)], 3, [(0, 12)]),
        ([(None, 2), (1, 5), (None, 1)], 3, [(1, 8)]),
        ([(None, 4)], 3, [(None, 4)]),
        ([(1, 1), (2, 1)], 3, [(2, 2)]),
        ([(0, 10), (1, 1), (0, 10)], 1, [(0, 10), (1, 1), (0, 10)]),
    ],
    ids=[
        "handoff_kept",
        "flicker_folded",
        "short_run_joins_longer_neighbor",
        "short_leading_run",
        "unmatched_frames_join_neighbors",
        "nothing_matches",
        "all_short_earliest_folds_first",
        "min_length_one_keeps_everything",
    ],
)
def test_fold_short_runs(runs, min_length, expected) -> None:
    """Short and unmatched runs fold into a neighbor; every frame stays covered."""
    from jabs.scripts.cli.label_remap_matching import _fold_short_runs

    folded = _fold_short_runs(runs, min_length)

    assert folded == expected
    assert sum(n for _, n in folded) == sum(n for _, n in runs)


def _handoff_poses(frames: int = 10, handoff: int = 5):
    """Source identity 0 is one animal; in the destination it is identity 1, then 2."""
    square = [_rect(0, 0, 19, 19)]
    src = _pose(masks={0: [square] * frames})
    dst = _pose(
        masks={
            0: [[_rect(100, 100, 119, 119)]] * frames,
            1: [square if f < handoff else None for f in range(frames)],
            2: [square if f >= handoff else None for f in range(frames)],
        }
    )
    return src, dst


def test_match_runs_splits_at_handoff() -> None:
    """A block whose animal changes destination identity is split at the change."""
    src, dst = _handoff_poses()

    runs = IdentityMatcher(src, dst, MatchSettings()).match_runs(0, 0, 9)

    assert [(r.start, r.end, r.match.identity) for r in runs] == [(0, 4, 1), (5, 9, 2)]
    assert all(r.match.accepted and r.match.score == pytest.approx(1.0) for r in runs)


def test_match_runs_without_splitting_uses_one_identity() -> None:
    """--no-split-blocks returns the single whole-block match."""
    src, dst = _handoff_poses()
    matcher = IdentityMatcher(src, dst, MatchSettings(split_blocks=False))

    [run] = matcher.match_runs(0, 0, 9)

    assert (run.start, run.end) == (0, 9)
    assert run.match == matcher.match(0, 0, 9)


def test_match_runs_folds_single_frame_flicker() -> None:
    """A one-frame switch to another identity does not split the block."""
    frames = 9
    square = [_rect(0, 0, 19, 19)]
    src = _pose(masks={0: [square] * frames})
    dst = _pose(
        masks={
            0: [None if f == 4 else square for f in range(frames)],
            1: [square if f == 4 else None for f in range(frames)],
        }
    )

    [run] = IdentityMatcher(src, dst, MatchSettings()).match_runs(0, 0, frames - 1)

    assert (run.start, run.end, run.match.identity) == (0, 8, 0)
    # the folded frame counts as 0 for identity 0, so the median stays 1.0 here
    assert run.match.score == pytest.approx(1.0)


def test_run_score_counts_missing_frames_as_zero() -> None:
    """A run's identity missing on most of its frames scores low and is rejected."""
    frames = 9
    square = [_rect(0, 0, 19, 19)]
    src = _pose(masks={0: [square] * frames})
    # destination identity 0 overlaps on only 3 of 9 frames; nothing else overlaps
    dst = _pose(
        masks={0: [square if f < 3 else [_rect(200, 200, 219, 219)] for f in range(frames)]}
    )
    settings = MatchSettings(min_run_frames=1)

    runs = IdentityMatcher(src, dst, settings).match_runs(0, 0, frames - 1)

    assert [(r.start, r.end) for r in runs] == [(0, 8)]
    assert runs[0].match.identity == 0
    assert runs[0].match.score == pytest.approx(0.0)
    assert not runs[0].match.accepted


def test_run_counts_folded_frames_without_destination_data_as_zero() -> None:
    """Frames folded in from a no-match run score 0, so a mostly missing run is rejected.

    The destination identity overlaps for 3 frames and is missing for the next 6; the
    6 frames fold into its run, and the run's median must reflect that it is absent.
    """
    frames = 9
    square = [_rect(0, 0, 19, 19)]
    src = _pose(masks={0: [square] * frames})
    dst = _pose(masks={0: [square if f < 3 else None for f in range(frames)]})

    [run] = IdentityMatcher(src, dst, MatchSettings()).match_runs(0, 0, frames - 1)

    assert (run.start, run.end, run.match.identity) == (0, 8, 0)
    assert run.match.score == pytest.approx(0.0)
    assert not run.match.accepted


def test_match_runs_with_no_destination_identities() -> None:
    """A destination pose with no identities gives one unmatched run instead of raising."""
    square = [_rect(0, 0, 19, 19)]
    src = _pose(masks={0: [square] * 5})
    dst = _pose(masks={})

    [run] = IdentityMatcher(src, dst, MatchSettings()).match_runs(0, 0, 4)

    assert (run.start, run.end, run.match.identity) == (0, 4, None)
    assert not run.match.accepted


def test_mixed_run_scores_bbox_only_frames_too() -> None:
    """In auto mode, box-only frames count toward a run's score, not just mask frames.

    The source has a mask on frames 0-3 only; on frames 4-9 only boxes can be compared,
    and the destination's box is far away. Those 6 frames must pull the run's score
    down instead of being ignored because the run also has mask frames.
    """
    frames = 10
    square = [_rect(0, 0, 19, 19)]
    near, far = _box(0, 0, 20, 20), _box(100, 100, 120, 120)
    src = _pose(
        boxes={0: np.stack([near] * frames)},
        masks={0: [square if f < 4 else None for f in range(frames)]},
    )
    dst = _pose(
        boxes={0: np.stack([near if f < 4 else far for f in range(frames)])},
        masks={0: [square] * frames},
    )

    [run] = IdentityMatcher(src, dst, MatchSettings()).match_runs(0, 0, frames - 1)

    assert (run.start, run.end, run.match.identity) == (0, 9, 0)
    assert run.match.metric is MatchMetric.SEGMENTATION
    assert run.match.score == pytest.approx(0.0)
    assert not run.match.accepted
