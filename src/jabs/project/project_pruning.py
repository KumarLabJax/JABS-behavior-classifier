from dataclasses import dataclass
from pathlib import Path

from jabs.pose_estimation import get_pose_path
from jabs.project import Project


@dataclass(frozen=True)
class VideoPaths:
    """A dataclass to hold paths related to a video to be pruned."""

    video_path: Path
    pose_path: Path
    annotation_path: Path


def get_videos_to_prune(project: Project, behavior: str | None = None) -> list[VideoPaths]:
    """Generates a list of videos that can be removed from the project due to lack of labels.

    Args:
        project (Project): The JABS project to check.
        behavior (str | None): The behavior to check for labels. If None, checks all behaviors.
    """

    def check_label_counts(label_counts: dict[str, dict[str, tuple[int, int]]]) -> bool:
        """Return True if any count in label_counts is greater than zero."""
        for identity_counts in label_counts.values():
            for counts in identity_counts.values():
                if any(count > 0 for count in counts):
                    return True
        return False

    videos_to_remove = []
    for video in project.video_manager.videos:
        video_path = project.video_manager.video_path(video)
        pose_path = get_pose_path(video_path)
        annotation_path = project.project_paths.annotations_dir / Path(video).with_suffix(".json")

        has_labels = False
        if behavior:
            counts = project.load_counts(video, behavior)
            has_labels = check_label_counts(counts)
        else:
            for b in project.settings_manager.behavior_names:
                counts = project.load_counts(video, b)
                has_labels = check_label_counts(counts)

                # found labels for at least one behavior, so we can stop checking
                if has_labels:
                    break

        # no labels for this video, so flag it for removal
        if not has_labels:
            videos_to_remove.append(VideoPaths(video_path, pose_path, annotation_path))

    return videos_to_remove
