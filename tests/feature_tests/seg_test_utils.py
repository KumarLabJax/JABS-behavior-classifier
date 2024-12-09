'''
This file contains the base class SegTest.  This class can be inherited by tests written that include segmentation 
data in the pose file (v6+).
'''

import h5py
import os
from pathlib import Path
import tempfile
import shutil
import gzip

import src.jabs.pose_estimation as pose_est


class SegDataBaseClass(object):
    _tmpdir = None
    data_dir = "data"
    dataPath = Path(__file__).parent.parent / data_dir
    dataFileName = "sample_pose_est_v6.h5.gz"

    @classmethod
    def setUpClass(cls):
        """
        This method overloads unittest.TestCase's setUp class level method.
        In this case the segmentation data reader can be reused.
        """

        # direct loading of segmentation data
        with h5py.File(os.path.join(cls.dataPath, cls.dataFileName), "r") as h5obj:
            cls.seg_data = h5obj.get("poseest/seg_data")[:]
        
        # create pose estimation v6 file, which also contains segmentation data
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tmpdir_path = Path(cls._tmpdir.name)

        open_func = open 

        try: 
            with gzip.open(cls.dataFileName, "rb") as fh:
                fh.read(1)
            open_func = gzip.open
        except OSError:
            """ Not a valid gun zip file. """

        with open_func(cls.dataPath / cls.dataFileName, 'rb') as f_in:
            with open(cls._tmpdir_path / cls.dataFileName.replace(".gz", ""), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        cls._pose_est_v6 = pose_est.open_pose_file(
            cls._tmpdir_path / cls.dataFileName.replace(".gz", ""))

    @ classmethod
    def tearDown(cls):
        if cls._tmpdir:
            cls._tmpdir.cleanup()


def setUpModule():
    """ Use if code should be executed once for all tests. """
    pass
