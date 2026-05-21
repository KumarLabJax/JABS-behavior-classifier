"""Update a JABS project in place to replace labels with labels remapped from another project.

This is the inverse of ``update-pose``: instead of keeping the target's labels and
replacing its pose, ``update-labels`` keeps the target's pose and replaces its
labels with labels imported from a source JABS project. The source provides both
labels and its own pose; the target's existing pose is the destination for the
identity-aware remap.

The script validates both projects, then performs the label remap in two
disposable staging projects so the live target stays unchanged while the update
is underway. Labels are processed block by block, matched by median bbox IoU
between the source pose and the target pose, and written directly to the staged
destination label track.

Timeline annotations are carried forward: video-level annotations are copied
as-is, and identity-scoped annotations are remapped by the same interval-matching
logic. If ``--drop-timeline-annotations`` is provided, source timeline
annotations are discarded instead of being copied or remapped.

Before the live target is modified, the script creates a timestamped backup zip
under ``<project>/.backup`` containing every live file the update may overwrite
or delete: ``jabs/project.json``, annotations, and predictions (pose files are
left untouched). If a failure occurs after the final live apply begins, the
script prints the backup path plus cleanup and manual restore instructions
instead of restoring automatically.

Behaviors named in the source project but not present in the target are merged
into the target's ``jabs/project.json`` so the imported labels are usable in the
GUI. Behaviors already configured in the target keep their existing
configuration. Existing target labels for videos that the source does not cover
are left untouched (per-video replace).

The target's pose is unchanged, so the feature cache stays valid and is not
regenerated. Predictions are cleared because they are stale relative to the new
labels; classifiers and the performance cache are left in place.

Example:
  jabs-cli update-labels /path/to/target_project /path/to/source_project --min-iou-thresh 0.5
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import click

from jabs.pose_estimation import get_pose_path
from jabs.project import Project

from .update_pose import (
    _copy_file_atomic,
    _create_backup_archive,
    _print_manual_restore,
    _project_videos,
    _restore_cleanup_paths,
    _run_staged_label_remap,
    _seed_stage_project,
    _validate_live_update_targets,
    _validate_pose_file,
)


def _source_annotated_videos(
    source_project_dir: Path,
    source_videos: set[str],
) -> list[str]:
    """Return source videos that have an annotation file, sorted by filename.

    Annotation files are matched by stem against the set of video filenames found
    in the source project. Source annotation files whose stem does not match a
    video in the source project are ignored with a warning.
    """
    annotations_dir = source_project_dir / "jabs" / "annotations"
    if not annotations_dir.is_dir():
        return []

    stem_to_video = {Path(v).stem: v for v in source_videos}
    annotated: list[str] = []
    for annotation_path in sorted(annotations_dir.glob("*.json")):
        video = stem_to_video.get(annotation_path.stem)
        if video is None:
            print(
                f"WARNING: Source annotation {annotation_path.name} has no matching "
                f"video file in {source_project_dir}; skipping.",
                file=sys.stderr,
            )
            continue
        annotated.append(video)
    return annotated


def _scan_target_for_init(target_dir: Path) -> list[str]:
    """Verify a non-initialized target directory looks like a JABS project before scaffolding.

    A target directory is acceptable for auto-init when it contains at least one
    video file and every video has a paired pose file. This is a stricter check
    than the basic "any video + any pose" gate in ``ProjectPaths.create_directories``
    and is meant to prevent silent scaffolding into an unrelated directory if the
    user mistyped the path.

    Args:
        target_dir: Directory expected to contain video and pose files.

    Returns:
        Sorted list of video filenames found in the directory.

    Raises:
        ValueError: If the directory has no videos, or any video is missing a paired
            pose file.
    """
    videos = _project_videos(target_dir)
    if not videos:
        raise ValueError(f"{target_dir} contains no video files (.avi or .mp4)")

    unpaired: list[str] = []
    for video in videos:
        try:
            pose_path = get_pose_path(target_dir / video)
        except ValueError:
            unpaired.append(video)
            continue
        if not pose_path.exists():
            unpaired.append(video)

    if unpaired:
        raise ValueError(f"{target_dir} is missing pose files for: {', '.join(unpaired)}")

    return videos


def _scaffold_target_project_if_missing(target_dir: Path) -> bool:
    """Scaffold a minimal JABS project in ``target_dir`` if it lacks ``jabs/project.json``.

    The target directory must already contain a self-consistent set of video and
    pose files (see ``_scan_target_for_init``) before any directories are created.
    Constructing :class:`Project` writes ``jabs/project.json`` with empty defaults
    and creates the standard subdirectory layout but does not compute features.

    Returns:
        ``True`` if the project was scaffolded, ``False`` if ``target_dir`` was
        already a valid JABS project.
    """
    if Project.is_valid_project_directory(target_dir):
        return False

    _scan_target_for_init(target_dir)
    print(
        f"INFO: {target_dir} has no jabs/ directory; initializing a minimal JABS project "
        "(features are not generated — run jabs-init separately to compute them).",
        file=sys.stderr,
    )
    Project(target_dir, enable_session_tracker=False)
    return True


def _preflight_label_update_inputs(
    target_dir: Path,
    source_project_dir: Path,
) -> tuple[list[str], set[str]]:
    """Validate target and source inputs, scaffolding the target if needed.

    The source is validated as a JABS project first, so an invalid source path
    cannot leave a partially initialized target on disk. Only after the source
    looks valid is ``target_dir`` either accepted (if it is already a JABS
    project) or scaffolded (if it is a videos + pose directory with no
    ``jabs/``). Scaffolding only creates an empty ``jabs/`` skeleton — no labels,
    features, or pose changes — so on any subsequent failure the user is left
    with a harmless no-op directory that the next run reuses.

    Returns:
        Tuple ``(videos, live_annotation_videos)`` where ``videos`` is the sorted
        list of source-labeled videos to operate on (each also exists in the
        target project) and ``live_annotation_videos`` is the subset of those
        videos that already have an annotation file in the live target.
    """
    if not Project.is_valid_project_directory(source_project_dir):
        raise ValueError(
            f"{source_project_dir} is not a valid JABS project directory (source labels)"
        )

    _scaffold_target_project_if_missing(target_dir)

    if not Project.is_valid_project_directory(target_dir):
        raise ValueError(f"{target_dir} is not a valid JABS project directory")

    target_videos = set(_project_videos(target_dir))
    if not target_videos:
        raise ValueError(f"No video files found in {target_dir}")

    source_videos = set(_project_videos(source_project_dir))
    if not source_videos:
        raise ValueError(f"No video files found in {source_project_dir}")

    labeled_videos = _source_annotated_videos(source_project_dir, source_videos)
    if not labeled_videos:
        raise ValueError(f"No labeled videos found in source project {source_project_dir}")

    missing_in_target = sorted(v for v in labeled_videos if v not in target_videos)
    if missing_in_target:
        raise ValueError(
            f"Source-labeled videos missing from target project: {', '.join(missing_in_target)}"
        )

    for video in labeled_videos:
        source_video_path = source_project_dir / video
        source_pose_path = get_pose_path(source_video_path)
        _validate_pose_file(
            video,
            source_video_path,
            source_pose_path,
            "source",
            require_bboxes=True,
        )

        target_video_path = target_dir / video
        target_pose_path = get_pose_path(target_video_path)
        _validate_pose_file(
            video,
            target_video_path,
            target_pose_path,
            "target",
            require_bboxes=True,
        )

    target_annotations_dir = target_dir / "jabs" / "annotations"
    live_annotation_videos: set[str] = set()
    if target_annotations_dir.exists():
        for video in labeled_videos:
            annotation_path = target_annotations_dir / Path(video).with_suffix(".json")
            if annotation_path.exists():
                live_annotation_videos.add(video)

    _validate_live_update_targets(
        target_dir,
        labeled_videos,
        live_annotation_videos,
        require_writable_pose_files=False,
        derived_dir_names=("predictions",),
    )

    return labeled_videos, live_annotation_videos


def _merge_source_behaviors_into_staged_project(
    source_project_dir: Path,
    dest_stage: Path,
) -> list[str]:
    """Merge behaviors from the source project's project.json into the staged target.

    Behaviors already present in the staged target keep their existing configuration;
    behaviors that exist only in the source are added with the source's configuration
    so the imported labels are usable in the GUI. Returns the list of behavior names
    newly added to the staged target.
    """
    source_project_file = source_project_dir / "jabs" / "project.json"
    staged_project_file = dest_stage / "jabs" / "project.json"

    try:
        source_settings = json.loads(source_project_file.read_text())
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Source project file {source_project_file} is not valid JSON: {exc}"
        ) from exc

    source_behaviors = source_settings.get("behavior")
    if not isinstance(source_behaviors, dict) or not source_behaviors:
        return []

    try:
        staged_settings = json.loads(staged_project_file.read_text())
    except FileNotFoundError:
        staged_settings = {}
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Staged project file {staged_project_file} is not valid JSON: {exc}"
        ) from exc

    staged_behaviors = staged_settings.get("behavior")
    if not isinstance(staged_behaviors, dict):
        staged_behaviors = {}

    newly_added: list[str] = []
    for name, settings in source_behaviors.items():
        if name in staged_behaviors:
            continue
        staged_behaviors[name] = settings
        newly_added.append(name)

    if not newly_added:
        return []

    staged_settings["behavior"] = staged_behaviors
    tmp = staged_project_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(staged_settings, indent=2, sort_keys=True))
    tmp.replace(staged_project_file)
    return sorted(newly_added)


def _apply_live_label_update(
    project_dir: Path,
    label_dest_project: Project,
    videos: list[str],
    backup_path: Path,
) -> None:
    """Apply staged label-update output back to the live target project.

    Copies staged annotations and ``project.json`` into the live target, then
    removes the live ``predictions/`` directory. Pose files, the feature/perf
    cache, classifiers, and features are left untouched.
    """
    live_annotations_dir = project_dir / "jabs" / "annotations"
    staged_annotations_dir = label_dest_project.project_paths.annotations_dir

    missing_annotations = sorted(
        video
        for video in videos
        if not (staged_annotations_dir / Path(video).with_suffix(".json")).exists()
    )
    if missing_annotations:
        raise RuntimeError(
            "staged label update did not produce annotations for: "
            + ", ".join(missing_annotations)
        )

    cleanup_paths = _restore_cleanup_paths(
        project_dir,
        videos,
        None,
        staged_annotations_dir,
    )

    try:
        live_annotations_dir.mkdir(parents=True, exist_ok=True)
        for video in videos:
            staged_annotation = staged_annotations_dir / Path(video).with_suffix(".json")
            if staged_annotation.exists():
                _copy_file_atomic(
                    staged_annotation,
                    live_annotations_dir / Path(video).with_suffix(".json"),
                )

        _copy_file_atomic(
            label_dest_project.project_paths.project_file,
            project_dir / "jabs" / "project.json",
        )

        shutil.rmtree(project_dir / "jabs" / "predictions", ignore_errors=True)
    except Exception as exc:
        print(
            f"ERROR: Failed while applying the label update to the live project: {exc}",
            file=sys.stderr,
        )
        _print_manual_restore(project_dir, backup_path, cleanup_paths)
        raise SystemExit(1) from exc


def update_project_labels_in_place(
    project_dir: Path,
    source_project_dir: Path,
    min_iou: float,
    verbose: bool = False,
    annotate_failures: bool = False,
    drop_timeline_annotations: bool = False,
) -> tuple[int, int, Path, list[str]]:
    """Update a live project in place by replacing its labels from ``source_project_dir``.

    The target's pose is left unchanged; incoming labels are remapped to the
    target's existing identity numbering via bbox IoU.

    Args:
        project_dir: Path to the live target project directory whose labels will be replaced.
        source_project_dir: Path to a JABS project providing the replacement labels and the
            pose used for IoU matching.
        min_iou: Minimum median IoU required to accept a block match.
        verbose: Whether to print successful block matches.
        annotate_failures: Whether to write timeline annotations for failed block matches.
        drop_timeline_annotations: Whether to discard source timeline annotations
            instead of copying or remapping them.

    Returns:
        Tuple ``(total_success, total_skipped, backup_path, newly_added_behaviors)``
        describing the completed update.
    """
    project_dir = project_dir.resolve()
    source_project_dir = source_project_dir.resolve()

    if project_dir == source_project_dir:
        raise ValueError("Target and source project directories must be different")

    labeled_videos, _live_annotation_videos = _preflight_label_update_inputs(
        project_dir,
        source_project_dir,
    )
    backup_path = _create_backup_archive(
        project_dir,
        labeled_videos,
        include_pose_files=False,
        prefix="update_labels",
    )

    with tempfile.TemporaryDirectory(prefix="jabs-update-labels-") as temp_root:
        temp_root_path = Path(temp_root)
        source_stage = temp_root_path / "source_stage"
        dest_stage = temp_root_path / "dest_stage"

        _seed_stage_project(source_stage, source_project_dir, copy_annotations=True)
        _seed_stage_project(dest_stage, project_dir, copy_annotations=False)

        newly_added_behaviors = _merge_source_behaviors_into_staged_project(
            source_project_dir,
            dest_stage,
        )

        label_source_project = Project(
            source_stage,
            enable_session_tracker=False,
            video_dir=source_project_dir,
            pose_dir=source_project_dir,
            validate_project_dir=False,
        )
        label_dest_project = Project(
            dest_stage,
            enable_session_tracker=False,
            video_dir=project_dir,
            pose_dir=project_dir,
            validate_project_dir=False,
        )

        total_success, total_skipped = _run_staged_label_remap(
            label_source_project,
            label_dest_project,
            min_iou,
            verbose,
            annotate_failures,
            drop_timeline_annotations,
            videos=labeled_videos,
            failure_description_phrase="label update",
        )

        _apply_live_label_update(
            project_dir,
            label_dest_project,
            labeled_videos,
            backup_path,
        )

    return total_success, total_skipped, backup_path, newly_added_behaviors


@click.command(
    name="update-labels",
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
    "source_project",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--min-iou-thresh",
    "min_iou",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum acceptable median IoU for a label remap match.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print successful label remap assignments in addition to warnings.",
)
@click.option(
    "--annotate-failures",
    is_flag=True,
    help="Add timeline annotations to the target for each block whose label remap fails.",
)
@click.option(
    "--drop-timeline-annotations",
    is_flag=True,
    help="Discard source timeline annotations instead of copying or remapping them.",
)
def update_labels_command(
    project: Path,
    source_project: Path,
    min_iou: float,
    verbose: bool,
    annotate_failures: bool,
    drop_timeline_annotations: bool,
) -> None:
    """Update a JABS project in place by replacing labels imported from another project."""
    total_success, total_skipped, backup_path, newly_added_behaviors = (
        update_project_labels_in_place(
            project,
            source_project,
            min_iou,
            verbose=verbose,
            annotate_failures=annotate_failures,
            drop_timeline_annotations=drop_timeline_annotations,
        )
    )

    click.echo(f"Backup archive: {backup_path}")
    click.echo(
        f"Label update summary: {total_success} label blocks assigned, "
        f"{total_skipped} label blocks skipped "
        f"(IoU threshold={min_iou})"
    )
    if newly_added_behaviors:
        click.echo(
            "Behaviors added to project.json from source: " + ", ".join(newly_added_behaviors)
        )
    else:
        click.echo("No new behaviors added to project.json.")
