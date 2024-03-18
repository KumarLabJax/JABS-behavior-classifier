import gzip
import json
import shutil
import unittest
from pathlib import Path

from src.project.project import Project
from src.project.video_labels import VideoLabels
from src.utils import hide_stderr


class TestProject(unittest.TestCase):
    """
    test project.project.Project

    TODO consider adding test for save_predictions method
    (save_predictions requires having pose_est file)
    """

    _EXISTING_PROJ_PATH = Path('test_project_with_data')
    _FILENAMES = ['test_file_1.avi', 'test_file_2.avi']

    # filenames of some compressed sample pose files in the test/data directory.
    # must be at least as long as _FILENAMES
    _POSE_FILES = ['identity_with_no_data_pose_est_v3.h5.gz',
                   'sample_pose_est_v3.h5.gz']

    @classmethod
    def setUpClass(cls):
        # create a project with empty video file and annotations

        test_data_dir = Path(__file__).parent / 'data'

        # make sure the test project dir is gone in case we previously
        # threw an exception during setup
        try:
            shutil.rmtree(cls._EXISTING_PROJ_PATH)
        except FileNotFoundError:
            pass

        cls._EXISTING_PROJ_PATH.mkdir()

        for i, name in enumerate(cls._FILENAMES):
            # make a stub for the .avi file in the project directory
            (cls._EXISTING_PROJ_PATH / name).touch()

            # extract the sample pose_est files
            pose_filename = name.replace('.avi', '_pose_est_v3.h5')
            pose_path = cls._EXISTING_PROJ_PATH / pose_filename
            pose_source = test_data_dir / cls._POSE_FILES[i]

            with gzip.open(pose_source, 'rb') as f_in:
                with open(pose_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        # setup a project directory with annotations
        Project(cls._EXISTING_PROJ_PATH, enable_video_check=False)

        # create an annotation
        labels = VideoLabels(cls._FILENAMES[0], 10000)
        walking_labels = labels.get_track_labels('0', 'Walking')
        walking_labels.label_behavior(100, 200)
        walking_labels.label_behavior(500, 1000)
        walking_labels.label_not_behavior(1001, 2000)

        # and manually place the .json file in the project directory
        with (cls._EXISTING_PROJ_PATH / 'jabs' / 'annotations' /
              Path(cls._FILENAMES[0]).with_suffix('.json')
        ).open('w', newline='\n') as f:
            json.dump(labels.as_dict(), f)

        # open project
        cls.project = Project(cls._EXISTING_PROJ_PATH, enable_video_check=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._EXISTING_PROJ_PATH)

    def test_create(self):
        """ test creating a new empty Project """
        project_dir = Path('test_project_dir')
        _ = Project(project_dir)

        # make sure that the empty project directory was created
        self.assertTrue(project_dir.exists())

        # make sure the jabs directory was created
        self.assertTrue((project_dir / Project._PROJ_DIR).exists())

        # make sure the jabs/annotations directory was created
        self.assertTrue(
            (project_dir / Project._PROJ_DIR / 'annotations').exists())

        # make sure the jabs/predictions directory was created
        self.assertTrue(
            (project_dir / Project._PROJ_DIR / 'predictions').exists())

        # remove project dir
        shutil.rmtree(project_dir)

    def test_get_video_list(self):
        """ get list of video files in an existing project """
        self.assertListEqual(self.project.videos, self._FILENAMES)

    def test_load_annotations(self):
        """ test loading annotations from a saved project """
        labels = self.project.load_video_labels(self._FILENAMES[0])

        with (self._EXISTING_PROJ_PATH / 'jabs' / 'annotations' /
              Path(self._FILENAMES[0]).with_suffix('.json')).open('r') as f:
            dict_from_file = json.load(f)

        self.assertTrue(len(self.project.videos), 2)

        # check to see that calling as_dict() on the VideoLabels object
        # matches what was used to load the annotation track from disk
        self.assertDictEqual(labels.as_dict(), dict_from_file)

    def test_save_annotations(self):
        """ test saving annotations """
        labels = self.project.load_video_labels(self._FILENAMES[0])
        walking_labels = labels.get_track_labels('0', 'Walking')

        # make some changes
        walking_labels.label_behavior(5000, 5500)

        # save changes
        self.project.save_annotations(labels)

        # make sure the .json file in the project directory matches the new
        # state
        with (self._EXISTING_PROJ_PATH / 'jabs' / 'annotations' /
              Path(self._FILENAMES[0]).with_suffix('.json')).open('r') as f:
            dict_from_file = json.load(f)

        self.assertDictEqual(labels.as_dict(), dict_from_file)

    def test_load_annotations_bad_filename(self):
        """
        test load annotations for a file that doesn't exist raises ValueError
        """
        with self.assertRaises(ValueError):
            self.project.load_video_labels('bad_filename.avi')

    def test_exception_creating_video_labels(self):
        """
        test OPError raised if unable to open avi file to get num frames
        """
        with self.assertRaises(IOError):
            with hide_stderr():
                self.project.load_video_labels(self._FILENAMES[1])

    def test_bad_video_file(self):
        with self.assertRaises(IOError):
            with hide_stderr():
                _ = Project(self._EXISTING_PROJ_PATH)

    def test_min_pose_version(self):
        # dummy project contains version 3 and 4 pose files
        # min should be 3
        self.assertEqual(self.project._min_pose_version, 3)

    def test_can_use_social_true(self):
        self.assertTrue(self.project.can_use_social_features)
