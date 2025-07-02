import enum
import shutil
from typing import TYPE_CHECKING

from jabs.pose_estimation import get_pose_path

if TYPE_CHECKING:
    from .project import Project


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


def merge_projects(destination: "Project", source: "Project", strategy: MergeStrategy) -> None:
    """Merges the source project into the destination project using the specified strategy.

    This function copies unique videos and pose files from the source to the destination,
    adds behaviors unique to the source project to the destination project, and merges
    or adds annotations for each video from the source project into the destination project
    according to the given merge strategy.

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
        destination_video_path = destination.project_paths.project_dir / source_video_path.name
        shutil.copy2(source_video_path, destination_video_path)

        # copy pose file
        source_pose_path = get_pose_path(source_video_path)
        destination_pose_path = destination.project_paths.project_dir / source_pose_path.name
        shutil.copy2(source_pose_path, destination_pose_path)

    # add behaviors that are unique to the source project to the destination project
    for behavior in behaviors_unique_to_source:
        print(f"Adding unique behavior {behavior} from source project to destination project...")
        destination.settings_manager.save_behavior(
            behavior, source.settings_manager.get_behavior(behavior)
        )

    # for each video in source, load the annotations and merge them into the destination project
    for video in source.video_manager.videos:
        source_labels = source.video_manager.load_video_labels(video)

        # if there are no labels for this video in the source project, skip it
        if source_labels is None:
            continue

        source_pose = source.load_pose_est(source.video_manager.video_path(video))
        destination_pose = destination.load_pose_est(destination.video_manager.video_path(video))

        if source_pose.hash != destination_pose.hash:
            print(f"WARNING: Pose hash mismatch for video {video}. Skipping annotation merge.")
            continue

        destination_labels = destination.video_manager.load_video_labels(video)

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
                continue
            except OSError as e:
                print(f"ERROR: Error saving annotations for video {video}: {e}")
                continue
