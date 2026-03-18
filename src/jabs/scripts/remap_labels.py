#!/usr/bin/env python

"""Remap behavior labels in place onto an updated pose set for a JABS project.

This is intended for keeping existing labels when pose files for the same
videos have been regenerated or otherwise updated. The script validates the
updated pose set first, then performs the remap in two disposable staging
projects so the live project stays unchanged while remapping is underway. The
updated pose directory must provide the same latest pose version for every
project video, and only that latest version is copied back into the live
project. Only after the staged remap succeeds does it copy the staged
annotations and project metadata back to the live project and swap in the
updated pose files.

Before the live project is modified, the script creates a timestamped backup zip
under ``<project>/.backup`` containing every live file the update may overwrite
or delete: pose files, annotations, ``jabs/project.json``, and predictions.
Cache is never backed up. If a failure occurs after the final live apply
begins, the script prints the backup path and manual restore instructions
instead of restoring automatically.

The matching behavior is unchanged from the original two-project workflow:
labels are processed block by block, matched by median bbox IoU, and written
directly to the destination label track with warnings on label overlap.

Example:
  python remap_labels.py /path/to/project /path/to/updated_pose_dir --min-iou-thresh 0.5
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from jabs.pose_estimation import get_pose_path, open_pose_file
from jabs.project import Project
from jabs.project.timeline_annotations import TimelineAnnotations
from jabs.project.video_labels import VideoLabels
from jabs.video_reader import VideoReader


def _project_videos(project_dir: Path) -> list[str]:
    """Return the sorted set of project video filenames."""
    return sorted(f.name for f in project_dir.glob("*") if f.suffix in {".avi", ".mp4"})


def _pose_files_for_video(video: str, pose_dir: Path) -> list[Path]:
    """Return all pose files in ``pose_dir`` that correspond to ``video``."""
    video_base = Path(video).with_suffix("").name
    return sorted(pose_dir.glob(f"{video_base}_pose_est_v*.h5"))


def _require_writable_directory(path: Path, description: str) -> None:
    """Raise if a directory does not permit the writes/removals needed by the live apply step."""
    if not path.exists():
        raise ValueError(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{description} is not a directory: {path}")
    if not os.access(path, os.W_OK | os.X_OK):
        raise PermissionError(f"{description} is not writable: {path}")


def _require_writable_existing_path(path: Path, description: str) -> None:
    """Raise if an existing file or directory is not writable."""
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(f"{description} is not writable: {path}")


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
    """Return the median bbox IoU across one source-identity/destination-identity interval.

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
        f"({same_count} identical, {conflict_count} conflicting frame(s)); identity mapping collision."
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
    track is updated immediately after each successful match, so later overlapping
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
    print(f"Saved remapped labels for {video} -> {dest_project.project_paths.annotations_dir}")
    return success_count, skipped_count


def _validate_pose_file(
    video: str,
    video_path: Path,
    pose_path: Path,
    role: str,
    require_bboxes: bool,
) -> int:
    """Validate that a pose file is readable and matches the video frame count."""
    pose = open_pose_file(pose_path, None)
    version = getattr(pose, "format_major_version", 0)
    if require_bboxes and (version < 8 or not getattr(pose, "has_bounding_boxes", False)):
        raise ValueError(f"{video} {role} pose v{version} lacks bounding boxes")

    pose_frames = pose.num_frames
    video_frames = VideoReader.get_nframes_from_file(video_path)
    if pose_frames != video_frames:
        raise ValueError(
            f"{video} {role} pose frame count ({pose_frames}) does not match video ({video_frames})"
        )

    return int(version)


def _validate_live_update_targets(
    project_dir: Path,
    videos: list[str],
    live_annotation_videos: set[str],
) -> None:
    """Best-effort preflight check that live update targets can be replaced or removed."""
    jabs_dir = project_dir / "jabs"
    annotations_dir = jabs_dir / "annotations"
    backup_dir = project_dir / ".backup"

    _require_writable_directory(project_dir, "live project directory")
    _require_writable_directory(jabs_dir, "live jabs directory")
    if annotations_dir.exists():
        _require_writable_directory(annotations_dir, "live annotations directory")

    if backup_dir.exists():
        _require_writable_directory(backup_dir, "backup directory")

    _require_writable_existing_path(jabs_dir / "project.json", "live project file")

    for video in videos:
        for live_pose_path in _pose_files_for_video(video, project_dir):
            _require_writable_existing_path(live_pose_path, "live pose file")

    for video in live_annotation_videos:
        annotation_path = annotations_dir / Path(video).with_suffix(".json")
        _require_writable_existing_path(annotation_path, "live annotation file")

    for derived_dir_name in ("predictions", "cache"):
        derived_dir = jabs_dir / derived_dir_name
        if not derived_dir.exists():
            continue

        _require_writable_directory(derived_dir, f"live {derived_dir_name} directory")
        for child in derived_dir.rglob("*"):
            if child.is_dir():
                _require_writable_directory(child, f"live {derived_dir_name} directory")
            else:
                _require_writable_existing_path(child, f"live {derived_dir_name} file")


def _preflight_remap_inputs(
    project_dir: Path,
    new_pose_dir: Path,
) -> tuple[list[str], dict[str, Path], set[str]]:
    """Validate live and replacement inputs before any live mutation occurs."""
    if not Project.is_valid_project_directory(project_dir):
        raise ValueError(f"{project_dir} is not a valid JABS project directory")
    if not new_pose_dir.is_dir():
        raise ValueError(f"{new_pose_dir} is not a directory")

    videos = _project_videos(project_dir)
    if not videos:
        raise ValueError(f"No video files found in {project_dir}")

    replacement_pose_files: dict[str, Path] = {}
    live_annotations: set[str] = set()
    annotations_dir = project_dir / "jabs" / "annotations"
    replacement_version: int | None = None

    for video in videos:
        video_path = project_dir / video
        live_pose_path = get_pose_path(video_path)
        replacement_pose_path = get_pose_path(video_path, new_pose_dir)

        _validate_pose_file(
            video,
            video_path,
            live_pose_path,
            "source",
            require_bboxes=True,
        )
        video_replacement_version = _validate_pose_file(
            video,
            video_path,
            replacement_pose_path,
            "replacement",
            require_bboxes=True,
        )

        if replacement_version is None:
            replacement_version = video_replacement_version
        elif video_replacement_version != replacement_version:
            raise ValueError(
                f"{video} replacement pose version v{video_replacement_version} does not match "
                f"replacement pose version v{replacement_version} used by other videos"
            )

        replacement_pose_files[video] = replacement_pose_path
        if (annotations_dir / Path(video).with_suffix(".json")).exists():
            live_annotations.add(video)

    _validate_live_update_targets(project_dir, videos, live_annotations)

    return videos, replacement_pose_files, live_annotations


def _create_backup_archive(project_dir: Path, videos: list[str]) -> Path:
    """Create a timestamped backup archive of live files that will be replaced or removed."""
    backup_dir = project_dir / ".backup"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"remap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    files_to_backup: set[Path] = set()
    for video in videos:
        files_to_backup.update(_pose_files_for_video(video, project_dir))

    project_file = project_dir / "jabs" / "project.json"
    if project_file.exists():
        files_to_backup.add(project_file)

    annotations_dir = project_dir / "jabs" / "annotations"
    if annotations_dir.exists():
        files_to_backup.update(path for path in annotations_dir.rglob("*") if path.is_file())

    predictions_dir = project_dir / "jabs" / "predictions"
    if predictions_dir.exists():
        files_to_backup.update(path for path in predictions_dir.rglob("*") if path.is_file())

    with zipfile.ZipFile(backup_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files_to_backup):
            zf.write(path, arcname=path.relative_to(project_dir))

    return backup_path


def _seed_stage_project(stage_root: Path, project_dir: Path, copy_annotations: bool) -> None:
    """Create a minimal staging project root seeded from the live project."""
    stage_jabs_dir = stage_root / "jabs"
    stage_annotations_dir = stage_jabs_dir / "annotations"
    stage_jabs_dir.mkdir(parents=True, exist_ok=True)
    stage_annotations_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(project_dir / "jabs" / "project.json", stage_jabs_dir / "project.json")

    if copy_annotations:
        live_annotations_dir = project_dir / "jabs" / "annotations"
        if live_annotations_dir.exists():
            shutil.copytree(live_annotations_dir, stage_annotations_dir, dirs_exist_ok=True)


def _refresh_project_identity_counts(project: Project) -> None:
    """Recompute the per-video identity counts stored in project.json."""
    video_metadata = project.settings_manager.project_settings.get("video_files", {})
    refreshed = dict(video_metadata)

    for video in project.video_manager.videos:
        vinfo = dict(refreshed.get(video, {}))
        pose = project.load_pose_est(project.video_manager.video_path(video))
        vinfo["identities"] = pose.num_identities
        refreshed[video] = vinfo

    project.settings_manager.save_project_file({"video_files": refreshed})


def _copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy a file into place via a temporary file and atomic replace."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, tmp_path)
    tmp_path.replace(destination)


def _print_manual_restore(project_dir: Path, backup_path: Path) -> None:
    """Print manual restore instructions for a previously created backup archive."""
    restore_zip = backup_path.relative_to(project_dir)
    command = f"cd {shlex.quote(str(project_dir))} && unzip -o {shlex.quote(str(restore_zip))}"
    print(f"Backup archive preserved at: {backup_path}", file=sys.stderr)
    print(f"To restore manually, run: {command}", file=sys.stderr)


def _apply_live_update(
    project_dir: Path,
    dest_project: Project,
    videos: list[str],
    replacement_pose_files: dict[str, Path],
    live_annotation_videos: set[str],
    backup_path: Path,
) -> None:
    """Apply staged remap output back to the live project."""
    live_annotations_dir = project_dir / "jabs" / "annotations"
    staged_annotations_dir = dest_project.project_paths.annotations_dir

    missing_annotations = sorted(
        video
        for video in live_annotation_videos
        if not (staged_annotations_dir / Path(video).with_suffix(".json")).exists()
    )
    if missing_annotations:
        raise RuntimeError(
            "staged remap did not produce annotations for: " + ", ".join(missing_annotations)
        )

    try:
        for video in videos:
            staged_annotation = staged_annotations_dir / Path(video).with_suffix(".json")
            if staged_annotation.exists():
                _copy_file_atomic(
                    staged_annotation,
                    live_annotations_dir / Path(video).with_suffix(".json"),
                )

        _copy_file_atomic(
            dest_project.project_paths.project_file, project_dir / "jabs" / "project.json"
        )

        for video in videos:
            for live_pose_path in _pose_files_for_video(video, project_dir):
                live_pose_path.unlink()

        for video in videos:
            replacement_pose_path = replacement_pose_files[video]
            _copy_file_atomic(replacement_pose_path, project_dir / replacement_pose_path.name)

        shutil.rmtree(project_dir / "jabs" / "predictions", ignore_errors=True)
        shutil.rmtree(project_dir / "jabs" / "cache", ignore_errors=True)
    except Exception as exc:
        print(
            f"ERROR: Failed while applying the remap to the live project: {exc}",
            file=sys.stderr,
        )
        _print_manual_restore(project_dir, backup_path)
        raise SystemExit(1) from exc


def _run_staged_remap(
    source_project: Project,
    dest_project: Project,
    min_iou: float,
    verbose: bool,
    annotate_failures: bool,
) -> tuple[int, int]:
    """Run the existing per-video remap semantics from source to destination."""
    total_success = 0
    total_skipped = 0

    for video in source_project.video_manager.videos:
        success, skipped = remap_labels_for_video(
            video,
            source_project,
            dest_project,
            min_iou,
            verbose=verbose,
            annotate_failures=annotate_failures,
        )
        total_success += success
        total_skipped += skipped

    return total_success, total_skipped


def remap_project_in_place(
    project_dir: Path,
    new_pose_dir: Path,
    min_iou: float,
    verbose: bool = False,
    annotate_failures: bool = False,
) -> tuple[int, int, Path]:
    """Remap a live project in place using replacement pose files from ``new_pose_dir``."""
    project_dir = project_dir.resolve()
    new_pose_dir = new_pose_dir.resolve()

    videos, replacement_pose_files, live_annotation_videos = _preflight_remap_inputs(
        project_dir,
        new_pose_dir,
    )
    backup_path = _create_backup_archive(project_dir, videos)

    with tempfile.TemporaryDirectory(prefix="jabs-remap-") as temp_root:
        temp_root_path = Path(temp_root)
        source_stage = temp_root_path / "source_stage"
        dest_stage = temp_root_path / "dest_stage"

        _seed_stage_project(source_stage, project_dir, copy_annotations=True)
        _seed_stage_project(dest_stage, project_dir, copy_annotations=False)

        source_project = Project(
            source_stage,
            enable_session_tracker=False,
            video_dir=project_dir,
            pose_dir=project_dir,
            validate_project_dir=False,
        )
        dest_project = Project(
            dest_stage,
            enable_session_tracker=False,
            video_dir=project_dir,
            pose_dir=new_pose_dir,
            validate_project_dir=False,
        )

        total_success, total_skipped = _run_staged_remap(
            source_project,
            dest_project,
            min_iou,
            verbose,
            annotate_failures,
        )
        _refresh_project_identity_counts(dest_project)
        _apply_live_update(
            project_dir,
            dest_project,
            videos,
            replacement_pose_files,
            live_annotation_videos,
            backup_path,
        )

    return total_success, total_skipped, backup_path


def main():
    """Main entry point for remapping project labels onto a replacement pose set."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("project", type=Path, help="Path to the live JABS project directory")
    parser.add_argument(
        "new_pose_dir",
        type=Path,
        help="Directory containing updated pose files for the project videos",
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
        help="Add timeline annotations to the project for each block that fails remap",
    )

    args = parser.parse_args()

    total_success, total_skipped, backup_path = remap_project_in_place(
        args.project,
        args.new_pose_dir,
        args.min_iou,
        verbose=args.verbose,
        annotate_failures=args.annotate_failures,
    )

    print(f"Backup archive: {backup_path}")
    print(
        f"Remap summary: {total_success} blocks assigned, {total_skipped} blocks skipped "
        f"(IoU threshold={args.min_iou})"
    )


if __name__ == "__main__":
    main()
