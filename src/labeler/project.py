from pathlib import Path
import json

from .video_labels import VideoLabels
from src.video_stream.utilities import get_frame_count


class Project:
    """ represents a labeling project """
    __PROJ_DIR = '.labeler'

    def __init__(self, project_path):
        """
        Open a project at a given path. A project is a directory that contains
        avi files with corresponding pose_est_v3.h5 files and json files
        containing project metadata and annotations.
        :param project_path: path to project directory
        """

        # make sure this is a pathlib.Path and not a string
        self._project_dir_path = Path(project_path)
        self._annotations_dir = self._project_dir_path / self.__PROJ_DIR / "annotations"

        # get list of video files in the project directory
        # TODO: we could check to see if the matching .h5 file exists
        self._videos = [f.name for f in self._project_dir_path.glob("*.avi")]
        self._videos.sort()

        # if project directory doesn't exist, create it (empty project)
        # parent directory must exist.
        Path(project_path).mkdir(mode=0o775, exist_ok=True)

        # make sure the project .labeler directory exists to store project
        # metadata and annotations
        Path(project_path, self.__PROJ_DIR).mkdir(mode=0o775, exist_ok=True)

        # make sure the project .labeler/annotations directory exists
        Path(project_path, self.__PROJ_DIR, "annotations").mkdir(
            mode=0o775, exist_ok=True)

        # unsaved annotations
        self._unsaved_annotations = {}

    @property
    def videos(self):
        """
        get list of video files that are in this project directory
        :return: list of file names (file names only, without path)
        """
        return self._videos

    def load_annotation_track(self, video_name):
        """
        load an annotation track from the project directory or from a cached of
        annotations that have previously been opened and not yet saved
        :param video_name: filename of the video: string or pathlib.Path
        :return: initialized VideoLabels object
        """

        video_filename = Path(video_name).name

        # make sure the video name actually matches one in the project
        if video_filename not in self._videos:
            raise ValueError(f"{video_filename} not in project")

        path = self._annotations_dir / Path(video_filename).with_suffix('.json')

        # if this has already been opened
        if video_filename in self._unsaved_annotations:
            annotations = self._unsaved_annotations.pop(video_filename)
            return VideoLabels.load(annotations)

        # if annotations already exist for this video file in the project open
        # it, otherwise create a new empty VideoLabels
        if path.exists():
            with path.open() as f:
                return VideoLabels.load(json.load(f))
        else:
            video_path = self._project_dir_path / video_filename
            nframes = get_frame_count(str(video_path))
            return VideoLabels(video_filename, nframes)

    def cache_unsaved_annotations(self, annotations):
        """
        Cache a VideoLabels object after encoding as a JSON serializable dict.
        Used when user switches from one video to another during a labeling
        project.
        :param annotations: VideoLabels object
        :return: None
        """
        self._unsaved_annotations[annotations.filename] = annotations.as_dict()

    def save_annotatios(self, annotations):
        """
        save state of a VideoLabels object to the project directory
        :param annotations: VideoLabels object
        :return: None
        """
        path = self._annotations_dir / Path(
            annotations.filename).with_suffix('.json')

        with path.open(mode='w', newline='\n') as f:
            json.dump(annotations.as_dict(), f)

    def save_cached_annotations(self):
        """
        save VideoLabel objects that have been cached
        :return: None
        """
        for video in self._unsaved_annotations:
            path = self._annotations_dir / Path(video).with_suffix('.json')

            with path.open(mode='w', newline='\n') as f:
                json.dump(self._unsaved_annotations[video], f)

    def video_path(self, video_file):
        """ take a video file name and generate the path used to open it """
        return Path(self._project_dir_path, video_file)
