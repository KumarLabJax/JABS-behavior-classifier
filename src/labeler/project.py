import glob
import os

class Project:
    def __init__(self, project_path):
        """
        Open a project at a given path. A project is a directory that contains
        avi files with corresponding pose_est_v3.h5 files and json files
        containing project metadata and annotations.
        :param project_path:
        """
        self._project_path = project_path
        self._vidoes = self._get_videos()

    def _get_videos(self):
        return [os.path.basename(f) for f in glob.glob(os.path.join(self._project_path, "*.avi"))]

    @property
    def videos(self):
        return self._vidoes

    def make_path(self, video_file):
        return os.path.join(self._project_path, video_file)
