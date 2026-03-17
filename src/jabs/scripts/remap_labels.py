#!/usr/bin/env python

"""Remap behavior labels between two JABS projects by matching identities with bbox IoU.

The script processes each labeled source block independently. For a given block,
it scores every destination identity by the median bounding-box IoU over the
block interval and selects the single best-scoring destination identity. This
is a per-block best match, not a greedy or one-to-one assignment across the
whole video, so multiple source identities may map to the same destination
identity.

Matches below the configured IoU threshold are skipped. Accepted blocks are
written directly into the destination label track. If a write overlaps labels
already written for the same destination identity and behavior, the script
warns. Identical overlaps are reported as mapping collisions, and conflicting
overlaps are also reported before the incoming block overwrites the previous
frame values. The saved label file therefore contains only the final resolved
per-frame labels, not overlapping contradictory intervals.

Both projects must use pose files with bounding boxes available.

Example:
  python remap_labels.py /path/to/source_proj /path/to/dest_proj --min-iou-thresh 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from jabs.project import Project
from jabs.project.timeline_annotations import TimelineAnnotations
from jabs.project.video_labels import VideoLabels


def _bboxes_for_identity(pose, identity: int) -> np.ndarray | None:
    """Return per-frame bboxes for an identity as [x1,y1,x2,y2], or None if unavailable."""
    if not hasattr(pose, "get_bounding_boxes") or not getattr(pose, "has_bounding_boxes", False):
        return None
    bboxes = pose.get_bounding_boxes(identity)
    if bboxes is None:
        return None
    # flatten [frames,2,2] -> [frames,4]
    return bboxes.reshape(bboxes.shape[0], 4)


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU for two boxes shaped (4,) = [x1,y1,x2,y2]."""
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


def interval_cost(
    src_pose,
    dst_pose,
    src_identity: int,
    dst_identity: int,
    start: int,
    end: int,
) -> float:
    """Return the median bbox IoU across the interval for one source/destination pair.

    Only frames where both boxes are finite and have positive area contribute to
    the score. If no such frames exist, returns 0.0.
    """
    src_b = _bboxes_for_identity(src_pose, src_identity)
    dst_b = _bboxes_for_identity(dst_pose, dst_identity)
    if src_b is None or dst_b is None:
        return 0.0

    src_slice = src_b[start : end + 1]
    dst_slice = dst_b[start : end + 1]

    # valid boxes: finite coords and positive width/height
    def _valid_boxes(boxes):
        finite = np.isfinite(boxes).all(axis=1)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        positive = (widths > 0) & (heights > 0)
        return finite & positive

    mask = _valid_boxes(src_slice) & _valid_boxes(dst_slice)
    if not np.any(mask):
        return 0.0

    ious = [_bbox_iou(a, b) for a, b in zip(src_slice[mask], dst_slice[mask], strict=True)]
    return float(np.median(ious)) if ious else 0.0


def find_best_identity(
    src_pose,
    dst_pose,
    src_identity: int,
    start: int,
    end: int,
) -> tuple[int | None, float]:
    """Find the single best destination identity for one source block interval.

    Destination identities are compared independently and the highest-IoU match
    is returned. The chosen identity is not reserved, so later blocks may match
    to the same destination identity.
    """
    best_id: int | None = None
    best_score = -float("inf")

    for dst_id in dst_pose.identities:
        iou = interval_cost(src_pose, dst_pose, src_identity, dst_id, start, end)
        if iou > best_score:
            best_score = iou
            best_id = dst_id

    return best_id, best_score


def _warn_on_label_overlap(
    video: str,
    behavior: str,
    src_identity: int,
    dst_identity: int,
    start: int,
    end: int,
    present: bool,
    dst_track_labels,
) -> None:
    """Warn when a converted block overlaps labels already written to the destination track.

    Reports both identical overlaps and conflicting overlaps for the same
    destination identity and behavior. Conflicting overlaps are still allowed;
    the later write replaces the previous frame values.
    """
    desired_label = (
        dst_track_labels.Label.BEHAVIOR if present else dst_track_labels.Label.NOT_BEHAVIOR
    )
    existing = dst_track_labels.get_labels()[start : end + 1]
    labeled = existing != dst_track_labels.Label.NONE
    if not np.any(labeled):
        return

    same_count = int(np.count_nonzero(existing[labeled] == desired_label))
    conflict_count = int(np.count_nonzero(existing[labeled] != desired_label))
    if same_count == 0 and conflict_count == 0:
        return

    message = (
        f"WARNING: {video} behavior={behavior} src_id={src_identity} -> dst_id={dst_identity} "
        f"frames={start}-{end} overlaps existing destination labels "
        f"({same_count} identical, {conflict_count} conflicting frame(s)); possible identity mapping collision."
    )
    if conflict_count:
        message += " Conflicting frames will be overwritten."
    print(message, file=sys.stderr)


def remap_labels_for_video(
    video: str,
    source_project: Project,
    dest_project: Project,
    min_iou: float,
    verbose: bool = False,
    annotate_failures: bool = False,
):
    """Remap labels for a single video.

    Source labels are read as contiguous blocks per source identity and
    behavior. Each block is matched independently to the destination identity
    with the highest median bbox IoU over that interval. The destination label
    track is updated immediately after each accepted match, so later overlapping
    writes for the same destination identity and behavior replace earlier frame
    values. Overlaps are warned before the write occurs.

    Returns:
        (success_count, skipped_count)
    """
    source_pose = source_project.load_pose_est(source_project.video_manager.video_path(video))
    dest_pose = dest_project.load_pose_est(dest_project.video_manager.video_path(video))

    # Require pose v8+ with bounding boxes
    for name, pose in (("source", source_pose), ("destination", dest_pose)):
        version = getattr(pose, "format_major_version", 0)
        if version < 8 or not getattr(pose, "has_bounding_boxes", False):
            print(
                f"WARNING: {video} {name} pose v{version} lacks bounding boxes; skipping.",
                file=sys.stderr,
            )
            return 0, 0

    if source_pose.num_frames != dest_pose.num_frames:
        print(
            f"WARNING: Frame count mismatch for {video} (source={source_pose.num_frames}, "
            f"dest={dest_pose.num_frames}). Skipping.",
            file=sys.stderr,
        )
        return 0, 0

    source_labels = source_project.video_manager.load_video_labels(video, pose=source_pose)
    if source_labels is None:
        print(f"INFO: No labels in source for {video}; skipping.")
        return 0, 0

    dest_labels = VideoLabels(video, dest_pose.num_frames)
    success_count = 0
    skipped_count = 0

    for src_identity_str, behaviors in source_labels._identity_labels.items():
        try:
            src_identity = int(src_identity_str)
        except ValueError:
            print(
                f"WARNING: Non-integer identity '{src_identity_str}' in {video}; skipping.",
                file=sys.stderr,
            )
            continue

        for behavior, track_labels in behaviors.items():
            for block in track_labels.get_blocks():
                start, end, present = block["start"], block["end"], block["present"]
                dst_identity, iou = find_best_identity(
                    source_pose, dest_pose, src_identity, start, end
                )

                if dst_identity is None or not np.isfinite(iou) or iou < min_iou:
                    print(
                        f"WARNING: {video} behavior={behavior} src_id={src_identity} frames={start}-{end} "
                        f"no match meeting IoU ≥ {min_iou:.2f} (best {iou:.2f}). Skipping block.",
                        file=sys.stderr,
                    )
                    skipped_count += 1

                    tag = "remap-behavior-failed" if present else "remap-not-behavior-failed"
                    if (
                        annotate_failures
                        and not dest_labels.timeline_annotations.annotation_exists(
                            start=start, end=end, tag=tag, identity_index=None
                        )
                    ):
                        dest_labels.timeline_annotations.add_annotation(
                            TimelineAnnotations.Annotation(
                                start=start,
                                end=end,
                                tag=tag,
                                color="#FF8800" if present else "#8888FF",
                                description=(
                                    f"remap failed: behavior={behavior}, present={present}, "
                                    f"src_id={src_identity}, best_iou={iou:.2f}"
                                ),
                                identity_index=None,
                            )
                        )
                    continue

                dst_track_labels = dest_labels.get_track_labels(str(dst_identity), behavior)
                _warn_on_label_overlap(
                    video,
                    behavior,
                    src_identity,
                    dst_identity,
                    start,
                    end,
                    present,
                    dst_track_labels,
                )
                if present:
                    dst_track_labels.label_behavior(start, end)
                else:
                    dst_track_labels.label_not_behavior(start, end)
                if verbose:
                    print(
                        f"MATCH: {video} behavior={behavior} src_id={src_identity} -> "
                        f"dst_id={dst_identity} frames={start}-{end} iou={iou:.2f}"
                    )
                success_count += 1

    dest_project.save_annotations(dest_labels, dest_pose)
    print(f"Saved converted labels for {video} -> {dest_project.project_paths.annotations_dir}")
    return success_count, skipped_count


def main():
    """Main entry point for converting JABS labels between projects."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source_project", type=Path, help="Path to source JABS project directory")
    parser.add_argument(
        "dest_project", type=Path, help="Path to destination JABS project directory"
    )
    parser.add_argument(
        "--min-iou-thresh",
        type=float,
        default=0.5,
        dest="min_iou",
        help="Minimum acceptable median IoU for a match (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print successful match assignments in addition to warnings",
    )
    parser.add_argument(
        "--annotate-failures",
        action="store_true",
        help="Add timeline annotations in destination for each block that fails conversion",
    )

    args = parser.parse_args()

    source_project = Project(args.source_project)
    dest_project = Project(args.dest_project)

    # Ensure destination project has the same behavior labels as the source.
    # This keeps label keys consistent even when the destination project is freshly created.
    source_behaviors = source_project.settings_manager.behavior_names
    dest_behaviors = set(dest_project.settings_manager.behavior_names)
    for behavior in source_behaviors:
        if behavior not in dest_behaviors:
            data = source_project.settings_manager.get_behavior(behavior)
            dest_project.settings_manager.save_behavior(behavior, data)

    source_videos = set(source_project.video_manager.videos)
    dest_videos = set(dest_project.video_manager.videos)

    missing_in_dest = source_videos - dest_videos
    if missing_in_dest:
        print(
            f"WARNING: Videos missing in destination: {', '.join(sorted(missing_in_dest))}",
            file=sys.stderr,
        )

    missing_in_source = dest_videos - source_videos
    if missing_in_source:
        print(
            f"WARNING: Videos missing in source: {', '.join(sorted(missing_in_source))}",
            file=sys.stderr,
        )

    common_videos = sorted(source_videos & dest_videos)
    if not common_videos:
        print("ERROR: No overlapping videos between projects.", file=sys.stderr)
        sys.exit(1)

    total_success = 0
    total_skipped = 0
    for video in common_videos:
        success, skipped = remap_labels_for_video(
            video,
            source_project,
            dest_project,
            args.min_iou,
            verbose=args.verbose,
            annotate_failures=args.annotate_failures,
        )
        total_success += success
        total_skipped += skipped

    print(
        f"Conversion summary: {total_success} blocks assigned, {total_skipped} blocks skipped "
        f"(IoU threshold={args.min_iou})"
    )


if __name__ == "__main__":
    main()
