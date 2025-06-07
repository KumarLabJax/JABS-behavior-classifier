import contextlib
import gzip
import json
import shutil
import unittest
from pathlib import Path
from typing import ClassVar

from src.jabs.project import Project, VideoLabels
from src.jabs.utils import hide_stderr


class TestProject(unittest.TestCase):
    """
    test project.project.Project

    TODO consider adding test for save_predictions method
    (save_predictions requires having pose_est file)
    """

    _EXISTING_PROJ_PATH = Path("test_project_with_data")
    _FILENAMES: ClassVar[list[str]] = ["test_file_1.avi", "test_file_2.avi"]

    # filenames of some compressed sample pose files in the test/data directory.
    # must be at least as long as _FILENAMES
    _POSE_FILES: ClassVar[list[str]] = [
        "identity_with_no_data_pose_est_v3.h5.gz",
        "sample_pose_est_v3.h5.gz",
    ]

    @classmethod
    def setUpClass(cls):
        """create a project with empty video file and annotations"""
        test_data_dir = Path(__file__).parent.parent / "data"

        # make sure the test project dir is gone in case we previously
        # threw an exception during setup
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(cls._EXISTING_PROJ_PATH)

        cls._EXISTING_PROJ_PATH.mkdir()

        for i, name in enumerate(cls._FILENAMES):
            # make a stub for the .avi file in the project directory
            (cls._EXISTING_PROJ_PATH / name).touch()

            # extract the sample pose_est files
            pose_filename = name.replace(".avi", "_pose_est_v3.h5")
            pose_path = cls._EXISTING_PROJ_PATH / pose_filename
            pose_source = test_data_dir / cls._POSE_FILES[i]

            with gzip.open(pose_source, "rb") as f_in, open(pose_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # set up a project directory with annotations
        Project(cls._EXISTING_PROJ_PATH, enable_video_check=False)

        # create an annotation
        labels = VideoLabels(cls._FILENAMES[0], 10000)
        walking_labels = labels.get_track_labels("0", "Walking")
        walking_labels.label_behavior(100, 200)
        walking_labels.label_behavior(500, 1000)
        walking_labels.label_not_behavior(1001, 2000)

        # and manually place the .json file in the project directory
        with (
            cls._EXISTING_PROJ_PATH
            / "jabs"
            / "annotations"
            / Path(cls._FILENAMES[0]).with_suffix(".json")
        ).open("w", newline="\n") as f:
            json.dump(labels.as_dict(), f)

        # open project
        cls.project = Project(cls._EXISTING_PROJ_PATH, enable_video_check=False)

    @classmethod
    def tearDownClass(cls):
        """remove the test project directory"""
        shutil.rmtree(cls._EXISTING_PROJ_PATH)

    def test_create(self):
        """test creating a new empty Project"""
        project_dir = Path("test_project_dir")
        project = Project(project_dir)

        # make sure that the empty project directory was created
        self.assertTrue(project_dir.exists())

        # make sure the jabs directory was created
        self.assertTrue(project.project_paths.jabs_dir.exists())

        # make sure the jabs/annotations directory was created
        self.assertTrue(project.project_paths.annotations_dir.exists())

        # make sure the jabs/predictions directory was created
        self.assertTrue(project.project_paths.prediction_dir.exists())

        # remove project dir
        shutil.rmtree(project_dir)

    def test_get_video_list(self):
        """get list of video files in an existing project"""
        self.assertListEqual(self.project.video_manager.videos, self._FILENAMES)

    def test_load_annotations(self):
        """test loading annotations from a saved project"""
        labels = self.project.video_manager.load_video_labels(self._FILENAMES[0])

        with (
            self._EXISTING_PROJ_PATH
            / "jabs"
            / "annotations"
            / Path(self._FILENAMES[0]).with_suffix(".json")
        ).open("r") as f:
            dict_from_file = json.load(f)

        self.assertTrue(len(self.project.video_manager.videos), 2)

        # check to see that calling as_dict() on the VideoLabels object
        # matches what was used to load the annotation track from disk
        self.assertDictEqual(labels.as_dict(), dict_from_file)

    def test_save_annotations(self):
        """test saving annotations"""
        labels = self.project.video_manager.load_video_labels(self._FILENAMES[0])
        walking_labels = labels.get_track_labels("0", "Walking")

        # make some changes
        walking_labels.label_behavior(5000, 5500)

        # save changes
        self.project.save_annotations(labels)

        # make sure the .json file in the project directory matches the new
        # state
        with (
            self._EXISTING_PROJ_PATH
            / "jabs"
            / "annotations"
            / Path(self._FILENAMES[0]).with_suffix(".json")
        ).open("r") as f:
            dict_from_file = json.load(f)

        # need to add the project labeler to the labels dict
        labels_as_dict = labels.as_dict()
        labels_as_dict["labeler"] = self.project.labeler

        self.assertDictEqual(labels_as_dict, dict_from_file)

    def test_load_annotations_bad_filename(self):
        """test load annotations for a file that doesn't exist raises ValueError"""
        with self.assertRaises(ValueError):
            self.project.video_manager.load_video_labels("bad_filename.avi")

    def test_no_saved_video_labels(self):
        """test loading labels for a video with no saved labels returns None"""
        assert self.project.video_manager.load_video_labels(self._FILENAMES[1]) is None

    def test_bad_video_file(self):
        """test loading a video file that doesn't exist raises ValueError"""
        with self.assertRaises(IOError), hide_stderr():
            _ = Project(self._EXISTING_PROJ_PATH)

    def test_min_pose_version(self):
        """dummy project contains version 3 and 4 pose files min should be 3"""
        self.assertEqual(self.project.feature_manager.min_pose_version, 3)

    def test_can_use_social_true(self):
        """test that can_use_social_features is True when social features are enabled"""
        self.assertTrue(self.project.feature_manager.can_use_social_features)
