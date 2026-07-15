import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from jabs.pose_estimation import (
    PoseEstimation,
    get_pose_path,
    open_pose_file,
)

from .project_paths import ProjectPaths
from .settings_manager import SettingsManager
from .video_labels import VideoLabels

if TYPE_CHECKING:
    from .parallel_workers import VideoScanResult

logger = logging.getLogger(__name__)


class VideoManager:
    """Manages video files and their associated metadata within a project.

    Handles initialization, validation, and metadata management for videos,
    including checking for corresponding pose files, verifying frame counts,
    and tracking the number of identities per video. Provides methods to
    retrieve video lists, load video labels, and access video-related
    information required for project processing.

    Args:
        paths: Object containing project directory paths.
        settings_manager: Manages project settings and metadata.
        enable_video_check: Opt-in up-front validation that video and pose file
            frame counts match; when True, every video file is opened. Defaults
            to False, which defers the check to when a video is opened.
        scan_results: Per-video metadata collected by
            :func:`~jabs.project.parallel_workers.scan_video_metadata`, keyed
            by video filename. Every video in the project directory must have an
            entry. Required fields for all callers: ``identity_count``,
            ``hdf5_frame_count``, ``static_objects``, ``lixit_keypoints``,
            ``has_cm_per_pixel``. When ``enable_video_check=True``,
            ``video_frame_count`` must also be populated (i.e. the scan job
            must have been run with ``scan_frame_counts=True``).
    """

    def __init__(
        self,
        paths: ProjectPaths,
        settings_manager: SettingsManager,
        enable_video_check: bool = False,
        *,
        scan_results: "dict[str, VideoScanResult]",
    ):
        self._paths = paths
        self._settings_manager = settings_manager
        self._videos = []
        self._video_identity_count = {}
        self._total_project_identities = 0
        self._pose_path_cache = {}  # Cache mapping video names to their pose file paths to avoid repeated lookups

        self._initialize_videos(enable_video_check, scan_results)

    def _initialize_videos(
        self,
        enable_video_check: bool,
        scan_results: "dict[str, VideoScanResult]",
    ) -> None:
        """Initialize video-related data and perform checks."""
        self._videos = self.get_videos(self._paths.video_dir)
        self._videos.sort()

        self._validate_pose_files()
        if enable_video_check:
            self._validate_video_frame_counts(scan_results)

        self._load_video_metadata(scan_results)

    @property
    def videos(self):
        """Get the list of video filenames in the project."""
        return self._videos

    @property
    def num_videos(self) -> int:
        """Get the number of videos in the project."""
        return len(self._videos)

    def remove_video(self, video_name: str):
        """Remove a video from the project.

        Args:
            video_name: Name of the video file to remove.
        """
        try:
            self.check_video_name(video_name)
        except ValueError as e:
            logger.warning("Error removing video %s: %s", video_name, e)
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
                    self.get_cached_pose_path(video_filename), self._paths.cache_dir
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
        """Get list of video filenames (without path) in a directory.

        Dotfiles are skipped. In particular this excludes macOS AppleDouble
        sidecar files (``._<name>``) that the OS creates when writing to
        non-APFS/HFS+ volumes (e.g. exFAT/NTFS external drives). Those are not
        real videos, and ``pathlib.Path.glob`` matches them (unlike the shell);
        their companion ``._..._pose_est_v*.h5`` sidecars are not valid HDF5 and
        would otherwise break the project video scan.
        """
        return [
            f.name
            for f in dir_path.glob("*")
            if f.suffix in [".avi", ".mp4"] and not f.name.startswith(".")
        ]

    def get_video_identity_count(self, video_name: str) -> int:
        """Get the number of identity count for a specific video.

        Args:
            video_name: Name of the video file

        Returns:
            Number of identities in the video
        """
        return self._video_identity_count.get(video_name, 0)

    def get_cached_pose_path(self, video_name: str) -> Path:
        """Get pose path for a video, using cache to avoid repeated lookups.

        Args:
            video_name: Name of the video file

        Returns:
            Path to the pose file

        Raises:
            ValueError: If video does not have a pose file
        """
        if video_name not in self._pose_path_cache:
            video_path = self.video_path(video_name)
            self._pose_path_cache[video_name] = get_pose_path(video_path, self._paths.pose_dir)
        return self._pose_path_cache[video_name]

    def _load_video_metadata(self, scan_results: "dict[str, VideoScanResult]") -> None:
        """Load identity counts from pre-scanned metadata.

        Args:
            scan_results: Per-video metadata from the project scan, keyed by
                video filename.
        """
        for video in self._videos:
            nidentities = scan_results[video]["identity_count"]
            self._video_identity_count[video] = nidentities
            self._total_project_identities += nidentities

    def _validate_video_frame_counts(self, scan_results: "dict[str, VideoScanResult]") -> None:
        """Ensure video and pose file frame counts match.

        Args:
            scan_results: Per-video metadata from the project scan, keyed by
                video filename. Must include ``hdf5_frame_count`` and
                ``video_frame_count`` (i.e. scanned with ``scan_frame_counts=True``).
        """
        err = False
        for v in self._videos:
            pose_frames = scan_results[v]["hdf5_frame_count"]
            vid_frames = scan_results[v]["video_frame_count"]
            if vid_frames is None:
                raise ValueError(
                    f"{v}: video_frame_count is missing from scan_results — "
                    "re-run the scan with scan_frame_counts=True"
                )
            if pose_frames != vid_frames:
                logger.error(
                    "%s: video and pose file have different number of frames (video=%s, pose=%s)",
                    v,
                    vid_frames,
                    pose_frames,
                )
                err = True
        if err:
            raise ValueError("Video and Pose File frame counts differ")

    def _validate_pose_files(self):
        """Ensure pose files exist for each video and populate pose path cache."""
        err = False
        for v in self.videos:
            try:
                # Populate cache during validation
                self._pose_path_cache[v] = get_pose_path(self.video_path(v), self._paths.pose_dir)
            except ValueError:
                logger.error("%s missing pose file", v)
                err = True
        if err:
            raise ValueError("Project missing pose file for one or more video")

    def video_path(self, video_file) -> Path:
        """take a video file name and generate the path used to open it"""
        return Path(self._paths.video_dir, video_file)

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
