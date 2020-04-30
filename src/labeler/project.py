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

        # make sure this is a pathlib.Path
        self._project_path = Path(project_path)

        # get list of video files in the project directory
        # TODO: we could check to see if the matching .h5 file exists
        self._videos = [f.name for f in self._project_path.glob("*.avi")]
        self._videos.sort()

        # make sure the project .labeler directory exists to store project
        # metadata and annotations
        Path(project_path, self.__PROJ_DIR).mkdir(mode=0o775, exist_ok=True)

        # make sure the project .labeler/annotations directory exists
        Path(project_path, self.__PROJ_DIR, "annotations").mkdir(
            mode=0o775, exist_ok=True)

        # unsaved annotations
        # could write these to a temp file on disk instead of keeping in memory
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
        :param video_name: filename of the video
        :return: initialized VideoLabels object
        """

        video_filename = Path(video_name).name
        filename = Path(video_name).with_suffix('.json')
        path = Path(self.__PROJ_DIR, "annotations", filename)

        if video_filename in self._unsaved_annotations:
            return VideoLabels.load(self._unsaved_annotations[video_filename])

        if path.exists():
            with path.open() as f:
                return VideoLabels.load(json.load(f))
        else:
            video_path = Path(
                self._project_path, video_filename)
            nframes = get_frame_count(str(video_path))
            return VideoLabels(video_filename, nframes)

    def cache_unsaved_annotations(self, annotations):
        self._unsaved_annotations[annotations.filename] = annotations.as_dict()

    def save_annotations(self):
        """
        TODO: this is a stub to finish in a another JIRA issue & pull request
        """
        raise NotImplementedError

    def video_path(self, video_file):
        """ take a video file name and generate the path used to open it """
        return Path(self._project_path, video_file)
