import enum
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from jabs.core.utils import hash_file
from jabs.pose_estimation import PoseOpenError, get_pose_path, open_pose_file

if TYPE_CHECKING:
    from .project import Project

logger = logging.getLogger(__name__)


class MergeStrategy(enum.IntEnum):
    """Enum for merge strategies when merging projects."""

    BEHAVIOR_WINS = enum.auto()
    NOT_BEHAVIOR_WINS = enum.auto()
    DESTINATION_WINS = enum.auto()


def unique_behaviors(project1: "Project", project2: "Project") -> tuple[set[str], set[str]]:
    """Find behaviors that are unique to each project."""
    behaviors1 = set(project1.settings_manager.behavior_names)
    behaviors2 = set(project2.settings_manager.behavior_names)
    unique_to_project1 = behaviors1 - behaviors2
    unique_to_project2 = behaviors2 - behaviors1

    return unique_to_project1, unique_to_project2


def unique_videos(project1: "Project", project2: "Project") -> tuple[set[str], set[str]]:
    """Find videos that are unique to each project."""
    videos1 = set(project1.video_manager.videos)
    videos2 = set(project2.video_manager.videos)
    unique_to_project1 = videos1 - videos2
    unique_to_project2 = videos2 - videos1

    return unique_to_project1, unique_to_project2


def _merge_annotations_with_pose(
    destination: "Project",
    source: "Project",
    video: str,
    strategy: MergeStrategy,
    copied: bool,
) -> None:
    """Merge one usable video's annotations from ``source`` into ``destination``.

    Args:
        destination: The project to merge into.
        source: The project to merge from.
        video: The video filename.
        strategy: How conflicting labels are resolved.
        copied: Whether the video was copied into ``destination`` by this merge.

    Raises:
        PoseOpenError: If either project's pose file cannot be opened. Nothing has
            been saved for the video when this is raised.
    """
    source_labels = source.video_manager.load_video_labels(video)

    # if there are no labels for this video in the source project, skip it
    if source_labels is None:
        return

    source_pose = source.load_pose_est(source.video_manager.video_path(video))
    if copied:
        # This video was just copied from source -> destination during this merge run.
        # Destination's in-memory VideoManager inventory has not been refreshed yet,
        # so check_video_name-based load paths may fail.
        try:
            destination_pose = destination.load_pose_est(
                destination.video_manager.video_path(video)
            )
        except ValueError:
            destination_video_path = destination.project_paths.video_dir / Path(video).name
            destination_pose = open_pose_file(
                get_pose_path(destination_video_path, destination.project_paths.pose_dir),
                destination.project_paths.cache_dir,
            )
        destination_labels = None
    else:
        destination_pose = destination.load_pose_est(destination.video_manager.video_path(video))
        destination_labels = destination.video_manager.load_video_labels(video)

    if source_pose.hash != destination_pose.hash:
        print(f"WARNING: Pose hash mismatch for video {video}. Skipping annotation merge.")
        return

    if destination_labels is None:
        # destination project does not have annotations for this video, so we can just copy the source annotations
        print(f"Adding annotations for video {video} to destination project...")
        destination.save_annotations(source_labels, destination_pose)
    else:
        # both projects have annotations for this video so need to merge source into destination
        print(
            f"Merging annotations for video {video} from source project into destination project..."
        )
        try:
            destination_labels.merge(source_labels, strategy)
            destination.save_annotations(destination_labels, destination_pose)
        except ValueError as e:
            print(f"ERROR: Error merging annotations for video {video}: {e}")
            return
        except OSError as e:
            print(f"ERROR: Error saving annotations for video {video}: {e}")
            return


def _unusable_reason(video: str, *projects: "Project") -> str | None:
    """Return why a video's pose file is unusable in any of the projects, or None."""
    for project in projects:
        if not project.is_usable(video):
            return project.unusable_videos[video]
    return None


def _merge_annotations_without_pose(
    destination: "Project",
    source: "Project",
    video: str,
    strategy: MergeStrategy,
    copied: bool,
) -> None:
    """Merge a video's annotation documents when its pose file cannot be opened.

    Annotations are JSON and do not need the pose: the documents are merged
    as :class:`VideoLabels` and saved without fragmenting the blocks by the
    identity mask (see :meth:`VideoLabels.as_dict`). External identities are
    carried over from the documents. As for a usable video, nothing is
    merged when the two pose files differ.

    Args:
        destination: The project to merge into.
        source: The project to merge from.
        video: The video filename.
        strategy: How conflicting labels are resolved.
        copied: Whether the video was copied into ``destination`` by this
            merge, so it has no annotations there yet and is not yet in
            ``destination``'s video manager.
    """
    # imported here: video_labels imports track_labels, which imports this module
    from .video_labels import VideoLabels

    source_document = source.video_manager.load_annotations(video)
    if source_document is None:
        return

    source_pose_path = source.video_manager.get_cached_pose_path(video)
    try:
        destination_pose_path = get_pose_path(
            destination.project_paths.video_dir / video, destination.project_paths.pose_dir
        )
        same_pose = hash_file(source_pose_path) == hash_file(destination_pose_path)
    except (OSError, ValueError) as e:
        # OSError: a pose file that cannot be read; ValueError: the destination
        # has no pose file for the video
        logger.warning(
            "Cannot read the pose file of video %s, so its annotations are not merged: %s",
            video,
            e,
        )
        return
    if not same_pose:
        logger.warning("Pose hash mismatch for video %s. Skipping annotation merge.", video)
        return

    destination_document = None if copied else destination.video_manager.load_annotations(video)
    labels = VideoLabels.load(source_document)
    if destination_document is None:
        logger.info("Adding annotations for video %s to destination project", video)
    else:
        logger.info(
            "Merging annotations for video %s from source project into destination project", video
        )
        destination_labels = VideoLabels.load(destination_document)
        destination_labels.merge(labels, strategy)
        labels = destination_labels

    document = labels.as_dict(
        None,
        project_metadata=destination.settings_manager.project_metadata,
        video_metadata=destination.settings_manager.video_metadata(video),
    )
    for existing in (destination_document, source_document):
        if existing is not None and "external_identities" in existing:
            document["external_identities"] = existing["external_identities"]
            break
    document["labeler"] = destination.labeler
    destination.annotation_store.save_document(video, document)
    destination.settings_manager.update_version()


def merge_projects(destination: "Project", source: "Project", strategy: MergeStrategy) -> None:
    """Merges the source project into the destination project using the specified strategy.

    This function copies unique videos and pose files from the source to the destination,
    adds behaviors unique to the source project to the destination project, and merges
    or adds annotations for each video from the source project into the destination project
    according to the given merge strategy. Annotations of a video whose pose
    file is unusable in either project, or cannot be opened, are merged without
    opening the pose (see :func:`_merge_annotations_without_pose`); a video whose
    pose file cannot be read at all is skipped with a warning.

    Args:
        destination (Project): The project to merge into (destination).
        source (Project): The project to merge from (source).
        strategy (MergeStrategy): The strategy to use when merging annotations.

    Returns:
        None
    """
    behaviors_unique_to_destination, behaviors_unique_to_source = unique_behaviors(
        destination, source
    )
    videos_unique_to_destination, videos_unique_to_source = unique_videos(destination, source)

    # copy videos and poses that are unique to the source project into the destination project
    for video in videos_unique_to_source:
        print(f"Copying unique video {video} from source project to destination project...")
        # Copy video file
        source_video_path = source.video_manager.video_path(video)
        destination_video_path = destination.project_paths.video_dir / source_video_path.name
        shutil.copy2(source_video_path, destination_video_path)

        # copy pose file
        source_pose_path = source.video_manager.get_cached_pose_path(video)
        destination_pose_path = destination.project_paths.pose_dir / source_pose_path.name
        shutil.copy2(source_pose_path, destination_pose_path)

    # add behaviors that are unique to the source project to the destination project
    for behavior in behaviors_unique_to_source:
        print(f"Adding unique behavior {behavior} from source project to destination project...")
        destination.settings_manager.save_behavior(
            behavior, source.settings_manager.get_behavior(behavior)
        )

    # for each video in source, load the annotations and merge them into the destination project
    for video in source.video_manager.videos:
        reason = _unusable_reason(video, source, destination)
        if reason is not None:
            logger.warning(
                "Merging annotations for %s without its pose file, which is unusable: %s",
                video,
                reason,
            )
            _merge_annotations_without_pose(
                destination, source, video, strategy, copied=video in videos_unique_to_source
            )
            continue

        copied = video in videos_unique_to_source
        try:
            _merge_annotations_with_pose(destination, source, video, strategy, copied)
        except PoseOpenError as e:
            # the scan accepted the pose, but it cannot be opened (e.g. the converter
            # refuses it): merge the annotations without it, as for an unusable video
            logger.warning(
                "Merging annotations for %s without its pose file, which cannot be opened: %s",
                video,
                e,
            )
            _merge_annotations_without_pose(destination, source, video, strategy, copied)
