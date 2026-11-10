"""Update a JABS project in place to use an updated pose set while remapping labels.

This is intended for keeping existing labels when pose files for the same
videos have been regenerated or otherwise updated. The script validates the
updated pose set first, then performs the label remap in two disposable
staging projects so the live project stays unchanged while the update is
underway. The updated pose directory must provide the same latest pose version
for every project video, and only that latest version is copied back into the
live project. Only after the staged label remap succeeds does it copy the
staged annotations and project metadata back to the live project and swap in
the updated pose files.

Before the live project is modified, the script creates a timestamped backup
zip under ``<project>/.backup`` containing every live file the update may
overwrite or delete: pose files, annotations, ``jabs/project.json``, and
predictions. If a failure occurs after the final live apply begins, the script
prints the backup path plus cleanup and manual restore instructions instead of
restoring automatically.

Labels are processed block by block and written directly to the destination
label track. Each block goes to the destination identity it overlaps best: by
segmentation mask IoU when both identities have masks, otherwise by bounding
box IoU (``--match-by`` changes this). Blocks are matched frame by frame and
split where the best-matching identity changes (``--no-split-blocks`` turns
this off).
Timeline annotations are also carried forward: video-level annotations are
copied as-is, and identity-scoped annotations are remapped by the same
interval-matching logic. If ``--drop-timeline-annotations`` is provided,
existing source timeline annotations are discarded instead of being copied or
remapped. After a successful live pose update, feature files are regenerated
automatically only when ``--skip-feature-gen`` is not passed and the existing
``jabs/project.json`` already contains explicit ``window_sizes``. If the
project file has no ``window_sizes`` entry, there is nothing to regenerate.
For more control over feature regeneration, use ``--skip-feature-gen`` and run
``jabs-init`` manually, or regenerate features from the GUI.

Example:
  jabs-cli update-pose /path/to/project /path/to/updated_pose_dir --min-iou-thresh 0.5
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import click
import h5py
import numpy as np

from jabs.core.utils import copy_file_atomic
from jabs.io.annotations import annotation_filename
from jabs.pose_estimation import get_pose_path, open_pose_file
from jabs.project import Project
from jabs.project.timeline_annotations import TimelineAnnotations
from jabs.project.video_labels import VideoLabels
from jabs.scripts.initialize_project import (
    DEFAULT_PROCESSES,
    run_initialize_project,
    validate_window_sizes,
)
from jabs.video_reader import VideoReader

from .label_remap_matching import (
    IdentityMatch,
    IdentityMatcher,
    MatchMetric,
    MatchSettings,
    RunMatch,
)
from .label_remap_options import identity_match_options, match_settings_from_options


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


def _threshold_text(match: IdentityMatch) -> str:
    """Describe the threshold a match was held to, e.g. ``mask IoU ≥ 0.50``."""
    name = "mask IoU" if match.metric is MatchMetric.SEGMENTATION else "bbox IoU"
    return f"{name} ≥ {match.threshold:.2f}"


def _match_outcome(match: IdentityMatch) -> str:
    """Describe what was done with a match, e.g. ``using segmentation -> dst_id=1``."""
    if match.accepted:
        return f"using {match.metric.value} -> dst_id={match.identity}"
    return f"no match accepted by {match.metric.value}"


def _warn_on_metric_disagreement(context: str, match: IdentityMatch) -> bool:
    """Warn when segmentation and bounding boxes pick different destination identities.

    Reported whether or not the match is accepted, so low-confidence disagreements
    are not hidden.

    Args:
        context: Description of the block or annotation being matched.
        match: The match.

    Returns:
        Whether the two scores disagreed.
    """
    if not match.disagreement:
        return False
    print(
        f"WARNING: {context} segmentation and bounding boxes disagree "
        f"({match.describe()}); {_match_outcome(match)}.",
        file=sys.stderr,
    )
    return True


def _annotate_disagreement(
    dest_labels: VideoLabels,
    start: int,
    end: int,
    match: IdentityMatch,
    failure_description_phrase: str,
) -> None:
    """Add a timeline annotation marking a block whose two match scores disagreed."""
    tag = "remap-metric-disagreement"
    if dest_labels.timeline_annotations.annotation_exists(
        start=start, end=end, tag=tag, identity_index=None
    ):
        return
    dest_labels.timeline_annotations.add_annotation(
        TimelineAnnotations.Annotation(
            start=start,
            end=end,
            tag=tag,
            color="#CC44CC",
            description=(
                f"segmentation and bounding boxes disagreed during {failure_description_phrase}: "
                f"{match.describe()}; {_match_outcome(match)}"
            ),
            identity_index=None,
        )
    )


# a block split into more parts than this is reported as ambiguous: the best match keeps
# changing, which usually means overlapping animals rather than a clean identity handoff
_AMBIGUOUS_SPLIT_PARTS = 4


def _is_ambiguous_split(runs: list[RunMatch]) -> bool:
    """Whether an interval was split into more parts than a clean handoff would produce."""
    return len(runs) > _AMBIGUOUS_SPLIT_PARTS


def _report_split(context: str, start: int, end: int, runs: list[RunMatch]) -> bool:
    """Print how an interval was split across destination identities, if it was.

    Splits into more than ``_AMBIGUOUS_SPLIT_PARTS`` parts also get a warning, since
    they usually mean overlapping animals and are worth reviewing.

    Args:
        context: Description of the block or annotation being matched.
        start: First frame of the interval.
        end: Last frame of the interval, inclusive.
        runs: The runs the interval was split into.

    Returns:
        Whether the interval was split.
    """
    if len(runs) < 2:
        return False
    parts = ", ".join(f"{r.start}-{r.end} -> dst_id={r.match.identity}" for r in runs)
    print(f"SPLIT: {context} frames={start}-{end} into {len(runs)} parts: {parts}")
    if _is_ambiguous_split(runs):
        print(
            f"WARNING: {context} frames={start}-{end} split into {len(runs)} parts; the best "
            "match keeps changing (likely overlapping animals). Review this block.",
            file=sys.stderr,
        )
    return True


def _annotate_split(
    dest_labels: VideoLabels, start: int, end: int, runs: list[RunMatch], context: str
) -> None:
    """Add a timeline annotation marking a block that was split across identities.

    Ambiguous splits (see `_is_ambiguous_split`) get their own tag so they can be
    reviewed separately from clean identity handoffs.
    """
    tag = "remap-ambiguous-split" if _is_ambiguous_split(runs) else "remap-split"
    if dest_labels.timeline_annotations.annotation_exists(
        start=start, end=end, tag=tag, identity_index=None
    ):
        return
    parts = ", ".join(f"{r.start}-{r.end} -> {r.match.identity}" for r in runs)
    dest_labels.timeline_annotations.add_annotation(
        TimelineAnnotations.Annotation(
            start=start,
            end=end,
            tag=tag,
            color="#22AAAA",
            description=f"label block split across identities: {context}: {parts}",
            identity_index=None,
        )
    )


def _add_timeline_annotation(
    dest_labels: VideoLabels, annotation: TimelineAnnotations.Annotation
) -> None:
    """Add ``annotation`` to ``dest_labels`` unless an identical one is already there."""
    if not dest_labels.timeline_annotations.annotation_exists(
        start=annotation.start,
        end=annotation.end,
        tag=annotation.tag,
        identity_index=annotation.identity_index,
    ):
        dest_labels.add_annotation(annotation)


def _remap_labels_for_video(
    video: str,
    label_source_project: Project,
    label_dest_project: Project,
    min_iou: float,
    verbose: bool = False,
    annotate_failures: bool = False,
    drop_timeline_annotations: bool = False,
    *,
    failure_description_phrase: str = "pose update",
    match_settings: MatchSettings | None = None,
):
    """Remap labels for a single video.

    Source labels are read as contiguous blocks per source identity and
    behavior. Each block is matched independently to the destination identity
    that overlaps it best: by segmentation mask IoU when both identities have
    masks, otherwise by bbox IoU (see `label_remap_matching.MatchMetric`). By
    default the match is made frame by frame and a block is split where its best
    destination identity changes, so an animal handed from one destination
    identity to another mid-block keeps its labels; each part is accepted or
    skipped on its own. When both scores are available and pick different
    identities, a warning is printed. The destination label
    track is updated immediately after each successful match, so later overlapping
    writes for the same destination identity and behavior replace earlier frame
    values. Overlaps are warned before the write occurs.

    Args:
        video: Video filename to remap.
        label_source_project: Project providing the current labels and source pose.
        label_dest_project: Staging project providing the replacement pose and output location.
        min_iou: Minimum median bbox IoU required to accept a bbox-decided match.
        verbose: Whether to print successful block matches.
        annotate_failures: Whether to add timeline annotations for failed block matches.
        drop_timeline_annotations: Whether to discard existing source timeline annotations
            instead of copying or remapping them.
        failure_description_phrase: Human-readable phrase used in the failure
            annotation description (e.g. ``"pose update"`` → "label remap failed
            during pose update: ..."). Both ``update-pose`` and ``update-labels``
            share the same tag names; this only affects the description text.
        match_settings: Matching options other than the bbox threshold, which comes
            from ``min_iou``. Defaults to `MatchSettings` defaults.

    Returns:
        Tuple of ``(success_count, skipped_count)``, counting each part of a split
        block separately.
    """
    source_pose = label_source_project.load_pose_est(
        label_source_project.video_manager.video_path(video)
    )
    dest_pose = label_dest_project.load_pose_est(
        label_dest_project.video_manager.video_path(video)
    )

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

    settings = replace(match_settings or MatchSettings(), min_bbox_iou=min_iou)
    matcher = IdentityMatcher(source_pose, dest_pose, settings)

    source_labels = label_source_project.video_manager.load_video_labels(video, pose=source_pose)
    if source_labels is None:
        print(f"INFO: No labels in source for {video}; skipping.")
        return 0, 0

    dest_labels = VideoLabels(video, dest_pose.num_frames)

    if not drop_timeline_annotations:
        for annotation in source_labels.timeline_annotations.serialize():
            start = annotation["start"]
            end = annotation["end"]
            tag = annotation["tag"]
            color = annotation["color"]
            description = annotation.get("description")
            src_annotation_identity = annotation.get("identity")

            if src_annotation_identity is None:
                # video-level annotations are copied as-is
                _add_timeline_annotation(
                    dest_labels,
                    TimelineAnnotations.Annotation(
                        start=start, end=end, tag=tag, color=color, description=description
                    ),
                )
            else:
                try:
                    src_annotation_identity = int(src_annotation_identity)
                except (TypeError, ValueError):
                    print(
                        f"WARNING: {video} annotation tag={tag} frames={start}-{end} has non-integer "
                        f"identity {src_annotation_identity!r}; skipping annotation.",
                        file=sys.stderr,
                    )
                    continue

                runs = matcher.match_runs(src_annotation_identity, start, end)
                _report_split(
                    f"{video} annotation tag={tag} src_id={src_annotation_identity}",
                    start,
                    end,
                    runs,
                )
                for run in runs:
                    match = run.match
                    context = (
                        f"{video} annotation tag={tag} src_id={src_annotation_identity} "
                        f"frames={run.start}-{run.end}"
                    )
                    if _warn_on_metric_disagreement(context, match) and annotate_failures:
                        _annotate_disagreement(
                            dest_labels, run.start, run.end, match, failure_description_phrase
                        )
                    if not match.accepted:
                        print(
                            f"WARNING: {context} no match meeting {_threshold_text(match)} "
                            f"({match.describe()}). Skipping annotation.",
                            file=sys.stderr,
                        )
                        continue
                    _add_timeline_annotation(
                        dest_labels,
                        TimelineAnnotations.Annotation(
                            start=run.start,
                            end=run.end,
                            tag=tag,
                            color=color,
                            description=description,
                            identity_index=match.identity,
                            display_identity=dest_pose.identity_index_to_display(match.identity),
                        ),
                    )

    success_count = 0
    skipped_count = 0

    for src_identity_str, behavior, track_labels in source_labels.iter_identity_behavior_labels():
        try:
            src_identity = int(src_identity_str)
        except ValueError:
            print(
                f"WARNING: Non-integer identity '{src_identity_str}' in {video}; skipping.",
                file=sys.stderr,
            )
            continue

        # Source's annotation references an identity its own pose can no longer
        # supply. Under normal strict operation this is caught earlier by
        # ``_handle_orphan_identities`` in preflight; this inline guard handles
        # the ``--tolerate-orphan-identities`` mode where preflight only warns.
        # All blocks for this (identity, behavior) pair are counted as skipped
        # with a single per-pair warning; no failure annotation is written
        # because the root cause is in the source project.
        if src_identity >= source_pose.num_identities:
            orphan_block_count = sum(1 for _ in track_labels.get_blocks())
            print(
                f"WARNING: {video} src_id={src_identity} not present in source pose "
                f"(only {source_pose.num_identities} identities); skipping "
                f"{orphan_block_count} block(s) for behavior={behavior}.",
                file=sys.stderr,
            )
            skipped_count += orphan_block_count
            continue

        for block in track_labels.get_blocks():
            present = block["present"]
            runs = matcher.match_runs(src_identity, block["start"], block["end"])
            block_context = f"{video} behavior={behavior} src_id={src_identity}"
            if (
                _report_split(block_context, block["start"], block["end"], runs)
                and annotate_failures
            ):
                _annotate_split(dest_labels, block["start"], block["end"], runs, block_context)

            for run in runs:
                start, end, match = run.start, run.end, run.match
                dst_identity = match.identity

                if (
                    _warn_on_metric_disagreement(f"{block_context} frames={start}-{end}", match)
                    and annotate_failures
                ):
                    _annotate_disagreement(
                        dest_labels, start, end, match, failure_description_phrase
                    )

                if not match.accepted:
                    print(
                        f"WARNING: {block_context} frames={start}-{end} "
                        f"no match meeting {_threshold_text(match)} ({match.describe()}). "
                        "Skipping block.",
                        file=sys.stderr,
                    )
                    skipped_count += 1

                    tag = "behavior-remap-failed" if present else "not-behavior-remap-failed"
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
                                    f"label remap failed during {failure_description_phrase}: "
                                    f"behavior={behavior}, present={present}, "
                                    f"src_id={src_identity}, {match.describe()}"
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
                        f"MATCH: {block_context} -> dst_id={dst_identity} frames={start}-{end} "
                        f"by {match.metric.value} ({match.describe()})"
                    )
                success_count += 1

    label_dest_project.save_annotations(dest_labels, dest_pose)
    print(
        f"Saved staged label remap for {video} -> {label_dest_project.project_paths.annotations_dir}"
    )
    return success_count, skipped_count


def _validate_pose_file(
    video: str,
    video_path: Path,
    pose_path: Path,
    role: str,
    require_bboxes: bool,
) -> tuple[int, int]:
    """Validate that a pose file is readable and matches the video frame count.

    Returns:
        Tuple of ``(format_major_version, num_identities)``.
    """
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

    return int(version), int(pose.num_identities)


def _orphan_identities_in_annotation(
    annotation_path: Path,
    num_identities: int,
) -> list[int]:
    """Return identity indices referenced in the annotation file that are orphaned.

    An identity index is orphaned when it is ``>= num_identities`` — i.e. the
    pose file does not have a corresponding row of pose data for it.

    Inspects the per-identity label tracks (from ``unfragmented_labels`` when
    present, else ``labels`` — mirroring :meth:`VideoLabels.load`, so whichever
    section the remap loop will actually iterate is the one that gets scanned)
    and identity-scoped timeline annotations (the optional ``identity`` field
    on each entry of ``annotations``). Non-integer or null identity values are
    ignored — they refer to video-level annotations or are otherwise out of
    scope for this check.

    Args:
        annotation_path: Path to a JABS annotation JSON file. Returns an empty
            list if the file does not exist or cannot be parsed.
        num_identities: The corresponding pose file's ``num_identities``. Any
            identity index ``>= num_identities`` is reported as orphaned.

    Returns:
        Sorted, deduplicated list of orphan identity indices.
    """
    try:
        with annotation_path.open() as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    orphans: set[int] = set()

    label_key = "unfragmented_labels" if "unfragmented_labels" in data else "labels"
    for identity_key in data.get(label_key, {}):
        try:
            identity_index = int(identity_key)
        except (TypeError, ValueError):
            continue
        if identity_index >= num_identities:
            orphans.add(identity_index)

    for annotation in data.get("annotations", []):
        identity = annotation.get("identity")
        if identity is None:
            continue
        try:
            identity_index = int(identity)
        except (TypeError, ValueError):
            continue
        if identity_index >= num_identities:
            orphans.add(identity_index)

    return sorted(orphans)


def _handle_orphan_identities(
    issues: list[tuple[str, int, list[int]]],
    *,
    source_label: str = "source",
    tolerate: bool = False,
) -> None:
    """Report orphan-identity issues, either by raising or by warning to stderr.

    Args:
        issues: List of ``(video, pose_num_identities, orphan_identity_indices)``
            tuples accumulated during preflight. Empty list is a no-op.
        source_label: Human label for the side that owns the broken data
            (e.g. ``"source"`` or ``"live"``). Used in the message.
        tolerate: When ``True``, write a consolidated warning to stderr and let
            the caller proceed. The remap loop's inline guard then skips any
            affected ``(identity, behavior)`` pairs as they are encountered.
            When ``False`` (default), raise a ``ValueError`` so the run aborts
            before any backup or mutation.
    """
    if not issues:
        return

    intro = (
        f"{source_label.capitalize()} project has orphan identity references "
        "that cannot be matched against pose data:"
    )
    lines = [intro]
    for video, num_identities, orphans in issues:
        orphans_str = ", ".join(str(i) for i in orphans)
        range_str = (
            "no valid indices"
            if num_identities == 0
            else f"valid indices 0 to {num_identities - 1}"
        )
        lines.append(
            f"  {video}: {source_label} pose has {num_identities} identities "
            f"({range_str}) but annotation references identity {orphans_str}"
        )
    lines.append("")
    lines.append(
        "These identity indices have no corresponding entries in the pose file. "
        "Typically the pose was regenerated with fewer identities, leaving stale "
        "label entries that can no longer be matched against pose data."
    )

    if tolerate:
        lines.append("")
        lines.append(
            "Proceeding because --tolerate-orphan-identities is set. The affected "
            "(identity, behavior) pairs will be skipped; all other labels are "
            "remapped normally."
        )
        print("WARNING: " + "\n".join(lines), file=sys.stderr)
        return

    lines.append("")
    lines.append(
        f"Clean up the {source_label} annotation files (drop the orphan identity "
        "entries from the affected .json files) and re-run, or pass "
        "--tolerate-orphan-identities to skip the affected entries instead."
    )
    raise ValueError("\n".join(lines))


def _validate_live_update_targets(
    project_dir: Path,
    videos: list[str],
    live_annotation_videos: set[str],
    *,
    require_writable_pose_files: bool = True,
    derived_dir_names: tuple[str, ...] = ("predictions", "cache"),
) -> None:
    """Best-effort preflight check that live update targets can be replaced or removed.

    Args:
        project_dir: Live project directory.
        videos: Video filenames that may need pose-file writability checks.
        live_annotation_videos: Subset of ``videos`` that have existing annotation files
            in the live project. These are checked for writability.
        require_writable_pose_files: When ``True`` (default), require that live pose files
            for every video are writable. Disable when the operation does not replace
            pose files (e.g. ``update-labels``).
        derived_dir_names: Derived jabs/ subdirectories that the apply step may remove.
            Each is checked for writability if it exists.
    """
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

    if require_writable_pose_files:
        for video in videos:
            for live_pose_path in _pose_files_for_video(video, project_dir):
                _require_writable_existing_path(live_pose_path, "live pose file")

    for video in live_annotation_videos:
        annotation_path = annotations_dir / Path(video).with_suffix(".json")
        _require_writable_existing_path(annotation_path, "live annotation file")

    for derived_dir_name in derived_dir_names:
        derived_dir = jabs_dir / derived_dir_name
        if not derived_dir.exists():
            continue

        _require_writable_directory(derived_dir, f"live {derived_dir_name} directory")
        for child in derived_dir.rglob("*"):
            if child.is_dir():
                _require_writable_directory(child, f"live {derived_dir_name} directory")
            else:
                _require_writable_existing_path(child, f"live {derived_dir_name} file")


class _VideoPoseCheck(NamedTuple):
    """Result of validating one video's live and replacement pose files."""

    video: str
    replacement_pose_path: Path
    live_num_identities: int
    replacement_version: int
    replacement_num_identities: int


def _check_video_pose_files(job: tuple[str, Path, Path]) -> _VideoPoseCheck:
    """Validate one video's live and replacement pose files (picklable worker function).

    Args:
        job: ``(video, project_dir, new_pose_dir)``.

    Returns:
        The replacement pose path, its version and identity count, and the live pose
        file's identity count.

    Raises:
        ValueError: If either pose file is missing, unreadable, lacks bounding boxes, or
            does not match the video's frame count.
    """
    video, project_dir, new_pose_dir = job
    video_path = project_dir / video
    live_pose_path = get_pose_path(video_path)
    replacement_pose_path = get_pose_path(video_path, new_pose_dir)
    _, live_num_identities = _validate_pose_file(
        video,
        video_path,
        live_pose_path,
        "source",
        require_bboxes=True,
    )
    replacement_version, replacement_num_identities = _validate_pose_file(
        video,
        video_path,
        replacement_pose_path,
        "replacement",
        require_bboxes=True,
    )
    return _VideoPoseCheck(
        video,
        replacement_pose_path,
        live_num_identities,
        replacement_version,
        replacement_num_identities,
    )


class _PreflightResult(NamedTuple):
    """Inputs validated by `_preflight_update_inputs`."""

    videos: list[str]
    replacement_pose_files: dict[str, Path]
    live_annotation_videos: set[str]
    replacement_identity_counts: dict[str, int]


def _preflight_update_inputs(
    project_dir: Path,
    new_pose_dir: Path,
    *,
    tolerate_orphan_identities: bool = False,
    processes: int = 1,
) -> _PreflightResult:
    """Validate live and replacement inputs before any live mutation occurs.

    Opening and checking each video's pose files is the slow part, so with
    ``processes`` > 1 it runs in parallel worker processes. Results are handled in
    video order, so the first problem reported is the same as with one process.

    Args:
        project_dir: Live project directory.
        new_pose_dir: Directory containing replacement pose files.
        tolerate_orphan_identities: When ``True``, orphan identity references in
            the live project's annotation files are warned about and remap
            continues, with the affected ``(identity, behavior)`` pairs skipped
            during remap. When ``False`` (default), the orphan check raises
            before any backup or mutation.
        processes: Number of worker processes used to check pose files.

    Returns:
        The project videos, each video's replacement pose file, the videos with live
        annotation files, and each replacement pose file's identity count.
    """
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
    orphan_issues: list[tuple[str, int, list[int]]] = []

    replacement_identity_counts: dict[str, int] = {}
    jobs = [(video, project_dir, new_pose_dir) for video in videos]
    with contextlib.ExitStack() as stack:
        if processes > 1 and len(jobs) > 1:
            pool = stack.enter_context(Pool(min(processes, len(jobs))))
            checks = pool.imap(_check_video_pose_files, jobs)
        else:
            checks = map(_check_video_pose_files, jobs)
        for check in checks:
            video = check.video
            live_num_identities = check.live_num_identities
            replacement_pose_path = check.replacement_pose_path
            video_replacement_version = check.replacement_version
            replacement_identity_counts[video] = check.replacement_num_identities
            if replacement_version is None:
                replacement_version = video_replacement_version
            elif video_replacement_version != replacement_version:
                raise ValueError(
                    f"{video} replacement pose version v{video_replacement_version} does not "
                    f"match replacement pose version v{replacement_version} used by other videos"
                )
            replacement_pose_files[video] = replacement_pose_path
            annotation_path = annotations_dir / Path(video).with_suffix(".json")
            if annotation_path.exists():
                live_annotations.add(video)
                orphans = _orphan_identities_in_annotation(annotation_path, live_num_identities)
                if orphans:
                    orphan_issues.append((video, live_num_identities, orphans))

    _handle_orphan_identities(
        orphan_issues, source_label="live", tolerate=tolerate_orphan_identities
    )

    _validate_live_update_targets(project_dir, videos, live_annotations)

    return _PreflightResult(
        videos, replacement_pose_files, live_annotations, replacement_identity_counts
    )


def _load_preexisting_window_sizes(project_dir: Path) -> tuple[int, ...]:
    """Load explicit window sizes from the existing project file.

    Missing ``window_sizes`` means there is nothing to regenerate.
    """
    project_file = project_dir / "jabs" / "project.json"
    if not project_file.exists():
        raise ValueError(f"{project_dir} is not a valid JABS project directory")

    try:
        project_settings = json.loads(project_file.read_text())
    except OSError as exc:
        raise ValueError(f"Unable to read project file {project_file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Project file {project_file} is not valid JSON: {exc}") from exc

    if "window_sizes" not in project_settings:
        return ()

    window_sizes = project_settings["window_sizes"]
    if not isinstance(window_sizes, list):
        raise ValueError(f"Project file {project_file} has invalid window_sizes data")

    try:
        return validate_window_sizes(window_sizes)
    except ValueError as exc:
        raise ValueError(f"Project file {project_file} has invalid window_sizes: {exc}") from exc


def _create_backup_archive(
    project_dir: Path,
    videos: list[str],
    *,
    include_pose_files: bool = True,
    prefix: str = "update_pose",
) -> Path:
    """Create a timestamped backup archive of live files that will be replaced or removed.

    Args:
        project_dir: Live project directory whose files should be backed up.
        videos: Video filenames whose pose files should be included when
            ``include_pose_files`` is true.
        include_pose_files: When ``True`` (default), include every per-video pose file
            in the archive. Disable when the operation does not replace pose files
            (e.g. ``update-labels``).
        prefix: Filename prefix for the generated archive. The full name is
            ``{prefix}_{YYYYMMDD_HHMMSS}.zip``.
    """
    backup_dir = project_dir / ".backup"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    files_to_backup: set[Path] = set()
    if include_pose_files:
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


def _refresh_project_identity_counts(
    project: Project, identity_counts: dict[str, int] | None = None
) -> None:
    """Recompute the per-video identity counts stored in project.json.

    Args:
        project: Project whose ``project.json`` is updated.
        identity_counts: Identity count of each video's pose file, if already known
            (e.g. from preflight). Videos not in it have their pose file opened.
    """
    video_metadata = project.settings_manager.project_settings.get("video_files", {})
    refreshed = dict(video_metadata)
    identity_counts = identity_counts or {}

    for video in project.video_manager.videos:
        vinfo = dict(refreshed.get(video, {}))
        if video in identity_counts:
            vinfo["identities"] = identity_counts[video]
        else:
            pose = project.load_pose_est(project.video_manager.video_path(video))
            vinfo["identities"] = pose.num_identities
        refreshed[video] = vinfo

    project.settings_manager.save_project_file({"video_files": refreshed})


def _load_pose_file_model_metadata(pose_path: Path) -> dict[str, object] | None:
    """Return parsed model metadata from a pose file, or ``None`` if it is absent."""
    try:
        with h5py.File(pose_path, "r") as pose_h5:
            raw_json = pose_h5["poseest"].attrs.get("model_metadata_json")
    except OSError as exc:
        raise ValueError(f"Could not read pose file metadata from {pose_path}: {exc}") from exc
    except KeyError as exc:
        raise ValueError(f"Pose file is missing the poseest group: {pose_path}") from exc

    if raw_json is None:
        return None
    if isinstance(raw_json, np.ndarray) and raw_json.shape == ():
        raw_json = raw_json.item()
    if isinstance(raw_json, bytes):
        raw_json = raw_json.decode("utf-8")
    if isinstance(raw_json, str):
        raw_json = raw_json.strip()
        if raw_json == "":
            return None

    try:
        metadata = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Pose file model metadata is not valid JSON in {pose_path}: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Pose file model metadata must be a JSON object in {pose_path}")

    return metadata


def _inject_consistent_pose_model_metadata(project: Project) -> dict[str, object] | None:
    """Merge shared replacement-pose model metadata into staged project metadata."""
    consistency_keys = (
        "pose_model_name",
        "pose_model_version",
        "detection_model_name",
        "detection_model_version",
    )
    metadata_by_video: dict[str, dict[str, object]] = {}
    missing_videos: list[str] = []

    for video in project.video_manager.videos:
        pose_path = project.video_manager.get_cached_pose_path(video)
        metadata = _load_pose_file_model_metadata(pose_path)
        if metadata is None:
            missing_videos.append(video)
            continue
        metadata_by_video[video] = metadata

    if not metadata_by_video:
        print(
            "WARNING: Replacement pose files are missing model_metadata_json; "
            "destination project metadata was not updated.",
            file=sys.stderr,
        )
        return None

    if missing_videos:
        present_videos = sorted(metadata_by_video)
        raise ValueError(
            "Replacement pose files have inconsistent model metadata presence: "
            f"missing for {', '.join(sorted(missing_videos))}; "
            f"present for {', '.join(present_videos)}"
        )

    first_video, first_metadata = next(iter(metadata_by_video.items()))
    missing_sentinel = object()
    for key in consistency_keys:
        first_value = first_metadata.get(key, missing_sentinel)
        for video, metadata in metadata_by_video.items():
            if metadata.get(key, missing_sentinel) != first_value:
                raise ValueError(
                    "Replacement pose files have inconsistent model metadata for "
                    f"{key}: {video} does not match {first_video}"
                )

    project.settings_manager.set_project_metadata({"project": first_metadata}, replace=False)
    return first_metadata


def _restore_cleanup_paths(
    project_dir: Path,
    videos: list[str],
    replacement_pose_files: dict[str, Path] | None,
    staged_annotations_dir: Path,
) -> list[Path]:
    """Return live-project files that may have been created during the failed apply step.

    Args:
        project_dir: Live project directory.
        videos: Videos that the apply step would touch.
        replacement_pose_files: Per-video replacement pose paths whose basenames may have
            been freshly written into ``project_dir``. Pass ``None`` (or omit entries) for
            operations that do not write new pose files (e.g. ``update-labels``).
        staged_annotations_dir: Directory containing staged annotation JSON files that
            may have been copied into the live annotations directory.
    """
    cleanup_paths: set[Path] = set()
    for video in videos:
        if replacement_pose_files is not None and video in replacement_pose_files:
            cleanup_paths.add(project_dir / replacement_pose_files[video].name)

        staged_annotation = staged_annotations_dir / Path(video).with_suffix(".json")
        if staged_annotation.exists():
            cleanup_paths.add(project_dir / "jabs" / "annotations" / staged_annotation.name)

    return sorted(cleanup_paths)


def _print_manual_restore(project_dir: Path, backup_path: Path, cleanup_paths: list[Path]) -> None:
    """Print cleanup and restore instructions for a previously created backup archive."""
    restore_zip = backup_path.relative_to(project_dir)
    print(f"Backup archive preserved at: {backup_path}", file=sys.stderr)

    if cleanup_paths:
        cleanup_targets = " ".join(
            shlex.quote(str(path.relative_to(project_dir))) for path in cleanup_paths
        )
        cleanup_command = f"cd {shlex.quote(str(project_dir))} && rm -f {cleanup_targets}"
        print(
            "Before restoring, remove files that may have been created during the failed apply:",
            file=sys.stderr,
        )
        print(cleanup_command, file=sys.stderr)

    restore_command = (
        f"cd {shlex.quote(str(project_dir))} && unzip -o {shlex.quote(str(restore_zip))}"
    )
    print(f"Then restore the backup archive: {restore_command}", file=sys.stderr)
    print(
        "If you prefer, restore into a clean copy of the project directory instead.",
        file=sys.stderr,
    )


def _apply_live_update(
    project_dir: Path,
    label_dest_project: Project,
    videos: list[str],
    replacement_pose_files: dict[str, Path],
    live_annotation_videos: set[str],
    backup_path: Path,
) -> None:
    """Apply staged pose-update output back to the live project."""
    live_annotations_dir = project_dir / "jabs" / "annotations"
    staged_annotations_dir = label_dest_project.project_paths.annotations_dir

    missing_annotations = sorted(
        video
        for video in live_annotation_videos
        if not (staged_annotations_dir / Path(video).with_suffix(".json")).exists()
    )
    if missing_annotations:
        raise RuntimeError(
            "staged pose update did not produce annotations for: " + ", ".join(missing_annotations)
        )

    cleanup_paths = _restore_cleanup_paths(
        project_dir,
        videos,
        replacement_pose_files,
        staged_annotations_dir,
    )

    try:
        for video in videos:
            staged_annotation = staged_annotations_dir / Path(video).with_suffix(".json")
            if staged_annotation.exists():
                copy_file_atomic(
                    staged_annotation,
                    live_annotations_dir / Path(video).with_suffix(".json"),
                )

        copy_file_atomic(
            label_dest_project.project_paths.project_file, project_dir / "jabs" / "project.json"
        )

        for video in videos:
            for live_pose_path in _pose_files_for_video(video, project_dir):
                live_pose_path.unlink()

        for video in videos:
            replacement_pose_path = replacement_pose_files[video]
            copy_file_atomic(replacement_pose_path, project_dir / replacement_pose_path.name)

        for derived_dir_name in ("predictions", "cache"):
            derived_dir = project_dir / "jabs" / derived_dir_name
            if derived_dir.exists():
                shutil.rmtree(derived_dir)
    except Exception as exc:
        print(
            f"ERROR: Failed while applying the pose update to the live project: {exc}",
            file=sys.stderr,
        )
        _print_manual_restore(project_dir, backup_path, cleanup_paths)
        raise SystemExit(1) from exc


def _regenerate_features_after_update(
    project_dir: Path,
    window_sizes: tuple[int, ...],
    backup_path: Path,
    skip_feature_gen: bool,
    processes: int = DEFAULT_PROCESSES,
) -> str:
    """Regenerate features after a successful pose update when requested by project settings."""
    if skip_feature_gen:
        return "skipped_by_option"

    if not window_sizes:
        return "skipped_no_window_sizes"

    try:
        run_initialize_project(
            force=False,
            processes=processes,
            window_sizes=validate_window_sizes(window_sizes),
            force_pixel_distances=False,
            metadata_path=None,
            skip_feature_generation=False,
            project_dir=project_dir,
        )
    except Exception as exc:
        print(
            f"ERROR: Pose update succeeded, but automatic feature regeneration failed: {exc}",
            file=sys.stderr,
        )
        print(f"Backup archive preserved at: {backup_path}", file=sys.stderr)
        print(
            "If you want to retry with more control, rerun update-pose with "
            "--skip-feature-gen and then run jabs-init manually, or regenerate "
            "features from the GUI.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    return "regenerated"


@dataclass(frozen=True)
class _RemapWorkerContext:
    """Arguments shared by every video in a parallel label remap."""

    label_source_project: Project
    label_dest_project: Project
    min_iou: float
    verbose: bool
    annotate_failures: bool
    drop_timeline_annotations: bool
    failure_description_phrase: str
    match_settings: MatchSettings | None


@dataclass(frozen=True)
class _RemapWorkerResult:
    """One video's remap counts and the output it printed."""

    success: int
    skipped: int
    stdout: str
    stderr: str


# set once in each worker process by _init_remap_worker
_worker_context: _RemapWorkerContext | None = None


def _init_remap_worker(context: _RemapWorkerContext) -> None:
    """Pool initializer: keep the shared remap arguments in the worker process."""
    global _worker_context
    _worker_context = context


def _remap_video_worker(video: str) -> _RemapWorkerResult:
    """Remap one video's labels in a worker process, capturing what it prints.

    The output is returned rather than printed so the parent can print each video's
    messages together, in video order.
    """
    context = _worker_context
    assert context is not None, "worker not initialized"
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        success, skipped = _remap_labels_for_video(
            video,
            context.label_source_project,
            context.label_dest_project,
            context.min_iou,
            verbose=context.verbose,
            annotate_failures=context.annotate_failures,
            drop_timeline_annotations=context.drop_timeline_annotations,
            failure_description_phrase=context.failure_description_phrase,
            match_settings=context.match_settings,
        )
    return _RemapWorkerResult(success, skipped, stdout.getvalue(), stderr.getvalue())


def _run_staged_label_remap(
    label_source_project: Project,
    label_dest_project: Project,
    min_iou: float,
    verbose: bool,
    annotate_failures: bool,
    drop_timeline_annotations: bool,
    *,
    videos: list[str] | None = None,
    failure_description_phrase: str = "pose update",
    match_settings: MatchSettings | None = None,
    processes: int = 1,
) -> tuple[int, int]:
    """Run the existing per-video label-remap semantics from source to destination.

    With ``processes`` > 1, videos are remapped in parallel worker processes. Each
    video's messages are printed together, in video order, once it finishes. Videos
    whose names differ only by extension share an annotation file; if any do, the
    videos are remapped one at a time so the result matches a serial run.

    Args:
        label_source_project: Source-side staged project providing labels and source pose.
        label_dest_project: Destination-side staged project receiving the remapped labels.
        min_iou: Minimum acceptable median IoU for a block match.
        verbose: Whether to print successful block matches.
        annotate_failures: Whether to write timeline annotations for failed block matches.
        drop_timeline_annotations: Whether to discard source timeline annotations
            instead of copying or remapping them.
        videos: Optional explicit list of video filenames to process. When ``None``
            (the default), iterate ``label_source_project.video_manager.videos``.
            Callers should pass the preflighted set of videos to ensure the source
            and destination projects can both load pose for every video processed.
        failure_description_phrase: Human-readable phrase used in the failure
            annotation description (e.g. ``"pose update"`` → "label remap failed
            during pose update: ..."). Both ``update-pose`` and ``update-labels``
            share the same tag names; this only affects the description text.
        match_settings: Matching options other than the bbox threshold (``min_iou``).
        processes: Number of worker processes; 1 remaps the videos one at a time in
            this process.
    """
    total_success = 0
    total_skipped = 0

    iteration_videos = videos if videos is not None else label_source_project.video_manager.videos

    parallel = processes > 1 and len(iteration_videos) > 1
    annotation_names = [annotation_filename(video) for video in iteration_videos]
    if parallel and len(set(annotation_names)) < len(annotation_names):
        # e.g. a.avi and a.mp4 both save to a.json; in parallel the last worker to
        # finish would win, so keep the serial order (the later video wins)
        print(
            "NOTE: some videos share an annotation file (same name, different extension); "
            "remapping videos one at a time.",
            file=sys.stderr,
        )
        parallel = False

    if parallel:
        # Saving annotations updates the app version in project.json when it differs.
        # Do that once here, so the workers' copies of the project are already current
        # and do not race to rewrite project.json.
        label_dest_project.settings_manager.update_version()
        context = _RemapWorkerContext(
            label_source_project,
            label_dest_project,
            min_iou,
            verbose,
            annotate_failures,
            drop_timeline_annotations,
            failure_description_phrase,
            match_settings,
        )
        with Pool(
            min(processes, len(iteration_videos)),
            initializer=_init_remap_worker,
            initargs=(context,),
        ) as pool:
            for result in pool.imap(_remap_video_worker, iteration_videos):
                sys.stdout.write(result.stdout)
                sys.stderr.write(result.stderr)
                total_success += result.success
                total_skipped += result.skipped
        return total_success, total_skipped

    for video in iteration_videos:
        success, skipped = _remap_labels_for_video(
            video,
            label_source_project,
            label_dest_project,
            min_iou,
            verbose=verbose,
            annotate_failures=annotate_failures,
            drop_timeline_annotations=drop_timeline_annotations,
            failure_description_phrase=failure_description_phrase,
            match_settings=match_settings,
        )
        total_success += success
        total_skipped += skipped

    return total_success, total_skipped


def update_project_pose_in_place(
    project_dir: Path,
    new_pose_dir: Path,
    min_iou: float,
    verbose: bool = False,
    annotate_failures: bool = False,
    drop_timeline_annotations: bool = False,
    skip_feature_gen: bool = False,
    tolerate_orphan_identities: bool = False,
    *,
    match_settings: MatchSettings | None = None,
    processes: int = 1,
) -> tuple[int, int, Path, str]:
    """Update a live project in place using replacement pose files from ``new_pose_dir``.

    Args:
        project_dir: Path to the live project directory to update.
        new_pose_dir: Directory containing replacement pose files.
        min_iou: Minimum median IoU required to accept a label remap match.
        verbose: Whether to print successful block matches.
        annotate_failures: Whether to write timeline annotations for failed block matches.
        drop_timeline_annotations: Whether to discard existing source timeline annotations
            instead of copying or remapping them.
        skip_feature_gen: Whether to skip feature generation after pose update.
        tolerate_orphan_identities: When ``True``, warn about orphan identity
            references in the live project's annotations and skip the affected
            ``(identity, behavior)`` pairs during remap instead of aborting at
            preflight.
        match_settings: Matching options other than the bbox threshold (``min_iou``).
        processes: Number of worker processes for the label remap and feature
            regeneration.

    Returns:
        Tuple of ``(total_success, total_skipped, backup_path, feature_regen_status)`` for
        the completed update.
    """
    project_dir = project_dir.resolve()
    new_pose_dir = new_pose_dir.resolve()
    preexisting_window_sizes = _load_preexisting_window_sizes(project_dir)

    videos, replacement_pose_files, live_annotation_videos, identity_counts = (
        _preflight_update_inputs(
            project_dir,
            new_pose_dir,
            tolerate_orphan_identities=tolerate_orphan_identities,
            processes=processes,
        )
    )
    backup_path = _create_backup_archive(project_dir, videos)

    with tempfile.TemporaryDirectory(prefix="jabs-update-pose-") as temp_root:
        temp_root_path = Path(temp_root)
        source_stage = temp_root_path / "source_stage"
        dest_stage = temp_root_path / "dest_stage"

        _seed_stage_project(source_stage, project_dir, copy_annotations=True)
        _seed_stage_project(dest_stage, project_dir, copy_annotations=False)

        label_source_project = Project(
            source_stage,
            enable_session_tracker=False,
            video_dir=project_dir,
            pose_dir=project_dir,
            validate_project_dir=False,
        )
        label_dest_project = Project(
            dest_stage,
            enable_session_tracker=False,
            video_dir=project_dir,
            pose_dir=new_pose_dir,
            validate_project_dir=False,
        )
        _inject_consistent_pose_model_metadata(label_dest_project)

        total_success, total_skipped = _run_staged_label_remap(
            label_source_project,
            label_dest_project,
            min_iou,
            verbose,
            annotate_failures,
            drop_timeline_annotations,
            match_settings=match_settings,
            processes=processes,
            videos=videos,
        )
        _refresh_project_identity_counts(label_dest_project, identity_counts)
        _apply_live_update(
            project_dir,
            label_dest_project,
            videos,
            replacement_pose_files,
            live_annotation_videos,
            backup_path,
        )

    feature_regen_status = _regenerate_features_after_update(
        project_dir,
        preexisting_window_sizes,
        backup_path,
        skip_feature_gen,
        processes=processes,
    )

    return total_success, total_skipped, backup_path, feature_regen_status


@click.command(
    name="update-pose",
    context_settings={"max_content_width": 120},
    help=__doc__,
)
@click.argument(
    "project",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.argument(
    "new_pose_dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@identity_match_options
@click.option(
    "-p",
    "--processes",
    "processes",
    type=click.IntRange(min=1),
    default=DEFAULT_PROCESSES,
    show_default=True,
    help=(
        "Number of worker processes used to remap videos in parallel"
        " and to regenerate features; defaults to the logical CPU count. 1 remaps videos one at a time."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print successful label remap assignments in addition to warnings.",
)
@click.option(
    "--annotate-failures",
    is_flag=True,
    help="Add timeline annotations to the project for each block whose label remap fails.",
)
@click.option(
    "--drop-timeline-annotations",
    is_flag=True,
    help="Discard existing source timeline annotations instead of copying or remapping them.",
)
@click.option(
    "--skip-feature-gen",
    is_flag=True,
    help=(
        "Skip automatic feature regeneration after a successful pose update. "
        "By default, features are regenerated only when project.json already "
        "contains explicit window_sizes."
    ),
)
@click.option(
    "--tolerate-orphan-identities",
    is_flag=True,
    help=(
        "Continue past orphan identity references in the live project's "
        "annotation files (warn instead of erroring at preflight). The affected "
        "(identity, behavior) pairs are skipped during remap; all other labels "
        "are remapped normally. By default, the command aborts at preflight."
    ),
)
def update_pose_command(
    project: Path,
    new_pose_dir: Path,
    min_iou: float,
    min_mask_iou: float,
    match_by: str,
    split_blocks: bool,
    min_run_frames: int,
    processes: int,
    verbose: bool,
    annotate_failures: bool,
    drop_timeline_annotations: bool,
    skip_feature_gen: bool,
    tolerate_orphan_identities: bool,
) -> None:
    """Update a JABS project in place to use updated pose files while remapping labels."""
    total_success, total_skipped, backup_path, feature_regen_status = update_project_pose_in_place(
        project,
        new_pose_dir,
        min_iou,
        match_settings=match_settings_from_options(
            min_iou=min_iou,
            min_mask_iou=min_mask_iou,
            match_by=match_by,
            split_blocks=split_blocks,
            min_run_frames=min_run_frames,
        ),
        processes=processes,
        verbose=verbose,
        annotate_failures=annotate_failures,
        drop_timeline_annotations=drop_timeline_annotations,
        skip_feature_gen=skip_feature_gen,
        tolerate_orphan_identities=tolerate_orphan_identities,
    )

    click.echo(f"Backup archive: {backup_path}")
    click.echo(
        f"Pose update summary: {total_success} label parts assigned, "
        f"{total_skipped} label parts skipped "
        f"(match by {match_by}, mask IoU threshold={min_mask_iou}, "
        f"bbox IoU threshold={min_iou}, "
        f"split blocks={'on' if split_blocks else 'off'})"
    )
    if feature_regen_status == "regenerated":
        click.echo("Feature regeneration: completed using window_sizes from jabs/project.json")
    elif feature_regen_status == "skipped_by_option":
        click.echo(
            "Feature regeneration: skipped (--skip-feature-gen). "
            "Run jabs-init manually or regenerate features from the GUI for more control."
        )
    else:
        click.echo(
            "Feature regeneration: skipped because jabs/project.json does not contain "
            "explicit window_sizes"
        )
