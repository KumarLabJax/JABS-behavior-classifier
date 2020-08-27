import unittest
import shutil
from pathlib import Path
import json

from src.labeler.project import Project
from src.labeler.video_labels import VideoLabels


class TestProject(unittest.TestCase):
    """ test labeler.project.Project """

    _EXISTING_PROJ_PATH = Path('test_project_with_data')
    _FILENAMES = ['test_file_1.avi', 'test_file_2.avi']

    @classmethod
    def setUpClass(cls):
        # create a project with a video file and annotations

        cls._EXISTING_PROJ_PATH.mkdir()
        for name in cls._FILENAMES:
            (cls._EXISTING_PROJ_PATH / name).touch()

        # create an empty project directory
        Project(cls._EXISTING_PROJ_PATH)

        # create an annotation
        labels = VideoLabels(cls._FILENAMES[0], 10000)
        walking_labels = labels.get_track_labels('0', 'Walking')
        walking_labels.label_behavior(100, 200)
        walking_labels.label_behavior(500, 1000)
        walking_labels.label_not_behavior(1001, 2000)

        # and manually place the .json file in the project directory
        with (cls._EXISTING_PROJ_PATH / 'rotta' / 'annotations' /
              Path(cls._FILENAMES[0]).with_suffix('.json')
        ).open('w', newline='\n') as f:
            json.dump(labels.as_dict(), f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._EXISTING_PROJ_PATH)

    def test_create(self):
        """ test creating a new empty Project """
        project_dir = Path('test_project_dir')
        proj = Project(project_dir)

        # make sure that the empty project directory was created
        self.assertTrue(project_dir.exists())

        # make sure the .labeler directory was created
        self.assertTrue((project_dir / 'rotta').exists())

        # make sure the .labeler/annotations directory was created
        self.assertTrue((project_dir / 'rotta' / 'annotations').exists())

        # remove project dir
        shutil.rmtree(project_dir)

    def test_get_video_list(self):
        """ get list of video files in an existing project """
        project = Project(self._EXISTING_PROJ_PATH)
        self.assertListEqual(project.videos, self._FILENAMES)

    def test_load_annotations(self):
        """ test loading annotations from a saved project """

        project = Project(self._EXISTING_PROJ_PATH)

        labels = project.load_annotation_track(self._FILENAMES[0])

        with (self._EXISTING_PROJ_PATH / 'rotta' / 'annotations' /
              Path(self._FILENAMES[0]).with_suffix('.json')).open('r') as f:
            dict_from_file = json.load(f)

        self.assertTrue(len(project.videos), 2)

        # check to see that calling as_dict() on the VideoLabels object
        # matches what was used to load the annotation track from disk
        self.assertDictEqual(labels.as_dict(), dict_from_file)

    def test_save_annotations(self):
        """ test saving annotations """
        project = Project(self._EXISTING_PROJ_PATH)
        labels = project.load_annotation_track(self._FILENAMES[0])
        walking_labels = labels.get_track_labels('0', 'Walking')

        # make some changes
        walking_labels.label_behavior(5000, 5500)

        # save changes
        project.save_annotations(labels)

        # make sure the .json file in the project directory matches the new
        # state
        with (self._EXISTING_PROJ_PATH / 'rotta' / 'annotations' /
              Path(self._FILENAMES[0]).with_suffix('.json')).open('r') as f:
            dict_from_file = json.load(f)

        self.assertDictEqual(labels.as_dict(), dict_from_file)

    def test_load_annotations_bad_filename(self):
        """
        test load annotations for a file that doesn't exist raises ValueError
        """
        project = Project(self._EXISTING_PROJ_PATH)

        with self.assertRaises(ValueError):
            labels = project.load_annotation_track('bad_filename.avi')

    def test_exception_creating_video_labels(self):
        """
        test OPError raised if unable to open avi file to get num frames
        """
        project = Project(self._EXISTING_PROJ_PATH)
        with self.assertRaises(IOError):
            pass
            labels = project.load_annotation_track(self._FILENAMES[1])
