import json
import sys
from pathlib import Path

from jabs.pose_estimation import (
    PoseEstimation,
    get_frames_from_file,
    get_pose_path,
    open_pose_file,
)
from jabs.video_reader import VideoReader

from .project_paths import ProjectPaths
from .settings_manager import SettingsManager
from .video_labels import VideoLabels


class VideoManager:
    """Manages video files and their associated metadata within a project.

    Handles initialization, validation, and metadata management for videos,
    including checking for corresponding pose files, verifying frame counts,
    and tracking the number of identities per video. Provides methods to
    retrieve video lists, load video labels, and access video-related
    information required for project processing.

    Args:
        paths (ProjectPaths): Object containing project directory paths.
        settings_manager (SettingsManager): Manages project settings and metadata.
        enable_video_check (bool, optional): Whether to validate video frame counts. Defaults to True.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        settings_manager: SettingsManager,
        enable_video_check: bool = True,
    ):
        self._paths = paths
        self._settings_manager = settings_manager
        self._videos = []
        self._video_identity_count = {}
        self._total_project_identities = 0

        self._initialize_videos(enable_video_check)

    def _initialize_videos(self, enable_video_check):
        """Initialize video-related data and perform checks."""
        self._videos = self.get_videos(self._paths.project_dir)
        self._videos.sort()

        self._validate_pose_files()
        if enable_video_check:
            self._validate_video_frame_counts()

        self._load_video_metadata()

    @property
    def videos(self):
        """Get the list of video filenames in the project."""
        return self._videos

    def remove_video(self, video_name: str):
        """Remove a video from the project.

        Args:
            video_name: Name of the video file to remove.
        """
        try:
            self.check_video_name(video_name)
        except ValueError as e:
            print(f"Error removing video {video_name}: {e}", file=sys.stderr)
        else:
            self._videos.remove(video_name)
            del self._video_identity_count[video_name]
            self._settings_manager.save_project_file()

    @property
    def total_project_identities(self) -> int:
        """Get the total number of identities across all videos."""
        return self._total_project_identities

    def load_video_labels(
        self, video_name: Path | str, pose: PoseEstimation = None
    ) -> VideoLabels | None:
        """load labels for a video

        Args:
            video_name: filename of the video: string or pathlib.Path
            pose: optional PoseEstimation object to use for identity mapping, if None we will open the pose file

        Returns:
            initialized VideoLabels object if annotations exist, otherwise None
        """
        video_filename = Path(video_name).name
        self.check_video_name(video_filename)

        path = self._paths.annotations_dir / Path(video_filename).with_suffix(".json")

        # if annotations already exist for this video file in the project open them
        if path.exists():
            # VideoLabels.load can use pose to convert identity index to the display identity
            if pose is None:
                pose = open_pose_file(
                    get_pose_path(self.video_path(video_filename)), self._paths.cache_dir
                )
            with path.open() as f:
                return VideoLabels.load(json.load(f), pose)
        else:
            return None

    def check_video_name(self, video_filename: str):
        """check that a video name matches one in the project

        Args:
            video_filename (str): name of the video file

        Returns:
            None

        Raises:
            ValueError: if the video is not in the project
        """
        if video_filename not in self._videos:
            raise ValueError(f"{video_filename} not in project")

    @staticmethod
    def get_videos(dir_path: Path):
        """Get list of video filenames (without path) in a directory"""
        return [f.name for f in dir_path.glob("*") if f.suffix in [".avi", ".mp4"]]

    def get_video_identity_count(self, video_name: str) -> int:
        """Get the number of identity count for a specific video.

        Args:
            video_name: Name of the video file

        Returns:
            Number of identities in the video
        """
        return self._video_identity_count.get(video_name, 0)

    def _load_video_metadata(self):
        """Load metadata for each video and calculate total identities."""
        video_metadata = self._settings_manager.project_settings.get("video_files", {})
        flush = False
        for video in self._videos:
            vinfo = video_metadata.get(video, {})
            nidentities = vinfo.get("identities")

            if not nidentities:
                pose_file = open_pose_file(
                    get_pose_path(self.video_path(video)), self._paths.cache_dir
                )
                nidentities = pose_file.num_identities
                vinfo["identities"] = nidentities
                flush = True

            self._video_identity_count[video] = nidentities
            self._total_project_identities += nidentities
            video_metadata[video] = vinfo
        if flush:
            self._settings_manager.save_project_file({"video_files": video_metadata})

    def _validate_video_frame_counts(self):
        """Ensure video and pose file frame counts match."""
        err = False
        for v in self._videos:
            path = get_pose_path(self.video_path(v))
            pose_frames = get_frames_from_file(path)
            vid_frames = VideoReader.get_nframes_from_file(self.video_path(v))
            if pose_frames != vid_frames:
                print(
                    f"{v}: video and pose file have different number of frames",
                    file=sys.stderr,
                )
                err = True
        if err:
            raise ValueError("Video and Pose File frame counts differ")

    def _validate_pose_files(self):
        """Ensure pose files exist for each video."""
        err = False
        for v in self.videos:
            if self._has_pose(v) is False:
                print(f"{v} missing pose file", file=sys.stderr)
                err = True
        if err:
            raise ValueError("Project missing pose file for one or more video")

    def _has_pose(self, vid: str) -> bool:
        """check to see if a video has a corresponding pose file"""
        path = self._paths.project_dir / vid

        try:
            get_pose_path(path)
        except ValueError:
            return False
        return True

    def video_path(self, video_file) -> Path:
        """take a video file name and generate the path used to open it"""
        return Path(self._paths.project_dir, video_file)

    def annotations_path(self, video_file) -> Path:
        """take a video file name and generate the path used to save associated annotations"""
        video_filename = Path(video_file).name
        self.check_video_name(video_filename)

        return self._paths.annotations_dir / Path(video_filename).with_suffix(".json")

    def load_annotations(self, video_file: str) -> dict | None:
        """Load annotations for a video file.

        Args:
            video_file: Name of the video file

        Returns:
            Annotations dictionary if it exists, otherwise None
        """
        path = self.annotations_path(video_file)
        if path.exists():
            with path.open() as f:
                return json.load(f)

        return None
