import json
import sys
from pathlib import Path

from jabs.pose_estimation import get_pose_path, get_frames_from_file, open_pose_file
from jabs.video_reader import VideoReader
from jabs.video_reader.utilities import get_frame_count
from .project_paths import ProjectPaths
from .settings_manager import SettingsManager
from .video_labels import VideoLabels



class VideoManager:

    def __init__(self, paths: ProjectPaths, settings_manager: SettingsManager, enable_video_check: bool = True):
        """
        Initialize the VideoManager with paths.

        :param paths: ProjectPaths object containing project directory paths
        """
        self._paths = paths
        self._settings_manager = settings_manager
        self._videos = []
        self._total_project_identities = 0

        self._initialize_videos(enable_video_check)

    def _initialize_videos(self, enable_video_check):
        """Initialize video-related data and perform checks."""
        self._videos = self.get_videos(self._paths.project_dir)
        self._videos.sort()

        #self._validate_pose_files()
        if enable_video_check:
            self._validate_video_frame_counts()

        self._load_video_metadata()

    @property
    def videos(self):
        return self._videos

    @property
    def total_project_identities(self) -> int:
        return self._total_project_identities

    def load_video_labels(self, video_name):
        """
        load labels for a video from the project directory or from a cached of
        annotations that have previously been opened and not yet saved
        :param video_name: filename of the video: string or pathlib.Path
        :return: initialized VideoLabels object
        """

        video_filename = Path(video_name).name
        self.check_video_name(video_filename)

        path = self._paths.annotations_dir / Path(video_filename).with_suffix('.json')

        # if annotations already exist for this video file in the project open
        # it, otherwise create a new empty VideoLabels
        if path.exists():
            with path.open() as f:
                return VideoLabels.load(json.load(f))
        else:
            video_path = self._paths.project_dir / video_filename
            nframes = get_frame_count(str(video_path))
            return VideoLabels(video_filename, nframes)

    def check_video_name(self, video_filename):
        """
        make sure the video name actually matches one in the project, this
        function will raise a ValueError if the video name is not valid,
        otherwise the function has no effect
        :param video_filename:
        :return: None
        :raises: ValueError if the filename is not a valid video in this project
        """
        if video_filename not in self._videos:
            raise ValueError(f"{video_filename} not in project")

    @staticmethod
    def get_videos(dir_path: Path):
        """ Get list of video filenames (without path) in a directory """
        return [f.name for f in dir_path.glob("*") if f.suffix in ['.avi', '.mp4']]

    def _load_video_metadata(self):
        """Load metadata for each video and calculate total identities."""
        video_metadata = self._settings_manager.project_settings.get('video_files', {})
        for video in self._videos:
            vinfo = video_metadata.get(video, {})
            nidentities = vinfo.get('identities')

            if nidentities is None:
                pose_file = open_pose_file(
                    get_pose_path(self.video_path(video)), self._paths.cache_dir)
                nidentities = pose_file.num_identities
                vinfo['identities'] = nidentities

            self._total_project_identities += nidentities
            video_metadata[video] = vinfo
        self._settings_manager.save_project_file({'video_files': video_metadata})

    def _validate_video_frame_counts(self):
        """Ensure video and pose file frame counts match."""
        err = False
        for v in self._videos:
            path = get_pose_path(self.video_path(v))
            pose_frames = get_frames_from_file(path)
            vid_frames = VideoReader.get_nframes_from_file(self.video_path(v))
            if pose_frames != vid_frames:
                print(f"{v}: video and pose file have different number of frames", file=sys.stderr)
                err = True
        if err:
            raise ValueError("Video and Pose File frame counts differ")

    def video_path(self, video_file) -> Path:
        """ take a video file name and generate the path used to open it """
        return Path(self._paths.project_dir, video_file)