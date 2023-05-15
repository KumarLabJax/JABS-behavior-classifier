'''
Author: Nick 
Purpose: This file was created because of the error thrown by test_pose_file.py.  
The tests here will be focussed on reading/ writing h5 files with gzip compression.
Conclusion: Real data is probably actually gzipped (adheres to file spec), whereas
test data is not.  The confusing thing is that h5py allows the user to gzip compress
individual datasets without actually making the h5 file a gzip file, so the .gz suffix
might be misleading.
'''

import unittest 
import numpy as np
import h5py 
import os 
import gzip
import shutil
from random import choice

class TestH5Data(unittest.TestCase):

    out_file = "tmp_data.h5"

    # some toy datasets to store in h5 files, can be overwritten in tests as needed.
    dataset = [
        ("d1", np.arange(10)),
        ("d2", np.arange(100)),
        ("d3", np.ones((1000,)))
    ]

    @classmethod
    def setUpClass(cls):
        di = np.arange(50)
        np.random.shuffle(di)
        cls.dataset.append((f"d{len(cls.dataset)+1}", di))

    def test_create_h5(self):
        '''Control, can create a simple h5 file with gzipped datasets.
        '''
        with h5py.File(self.out_file, "w") as h5obj:
            for name, data in self.dataset:
                h5obj.create_dataset(name, data=data, compression="gzip")

            for name, data in self.dataset:
                assert name in h5obj.keys()
        
        os.remove(self.out_file)
    
    def test_load_h5(self):
        '''Check the difference in file size between gzipped and non gzipped file.  Also try loading the gzipped file.
        '''
        compressed_file = f"compressed_{self.out_file}.gz" # adding a .gz suffix to the compressed dataset file.

        # write two h5 files with the same data, one file is compressed with gzip
        # the other is not.

        with h5py.File(compressed_file, "w") as h5obj_out_comp:
            for name, data in self.dataset:
                # also the chunks flag has been set
                # compression_opts sets the compression depth.
                h5obj_out_comp.create_dataset(name, data=data, compression="gzip", chunks=True, compression_opts=9)
        
        with h5py.File(self.out_file, "w") as h5obj_out:
            for name, data in self.dataset:
                h5obj_out.create_dataset(name, data=data)

        #  test ability to read h5 file with no compression
        with h5py.File(self.out_file, "r") as h5obj_in:
            for name, data in self.dataset:
                read_data = h5obj_in.get(name)[:]
                assert len(data) > 0 and len(read_data) == len(data)
                assert read_data[0] == data[0] and read_data[-1] == data[-1]
        
        #  test ability to read h5 file with compression & chunks
        with h5py.File(compressed_file, "r") as h5obj_in:
            for name, data in self.dataset:
                read_data = h5obj_in.get(name)[:]
                assert len(data) > 0 and len(read_data) == len(data)
                assert read_data[0] == data[0] and read_data[-1] == data[-1]

        # compare file sizes
        osz = os.path.getsize(self.out_file)
        csz = os.path.getsize(compressed_file)
        
        print("file sizes:",csz, osz, csz < osz)
        # compressed file size should be less than non compressed right?
        # fails if data is not highly repetitive.
        assert csz < osz

        # clean up temporary files
        os.remove(self.out_file)
        os.remove(compressed_file)
    

    def test_gzip_open(self):
        '''Trying to replicate the strategy employed in test_pose_file.
        '''
        compressed_file = f"compressed_{self.out_file}.gz" # adding a .gz suffix to the compressed dataset file.

        # write two h5 files with the same data, one file is compressed with gzip
        # the other is not.

        with h5py.File(compressed_file, "w") as h5obj_out_comp:
            for name, data in self.dataset:
                # also the chunks flag has been set
                # compression_opts sets the compression depth.
                h5obj_out_comp.create_dataset(name, data=data, compression="gzip", chunks=True, compression_opts=9)
    
        # Why is this being done?  The data can be directly read with h5py, no unzipping necessary.  Furthermore 
        # opening with gzip throws an error, whereas simply using the builtin open function throws no error.
        # Answer: Brian says additional level of gzip added to make files smaller, production data does not
        # have extra gzipping.

        open_func = open # gzip.open

        with open_func(compressed_file, "rb") as f_in:
            with open(compressed_file.replace('.h5.gz', '.h5'),'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        assert True
        
        # if open_func = gzip.open ->
        # ERROR: throws gzip.BadGzipFile: Not a gzipped file (b'\x89H', same error as test_pose_file

        # verify data integrity after file type conversion.
        with h5py.File(compressed_file.replace('.h5.gz', '.h5'), "r") as h5obj:
            name, data = choice(self.dataset)
            assert h5obj.get(name)[:][-1] == data[-1]
    
        os.remove(compressed_file.replace('.h5.gz', '.h5'))

    def test_extended_gzip(self):
        ''' In this test the gzip module is used to add an additional layer
        of gzip compression to the h5 python file.
        '''

        compressed_file = f"compressed_{self.out_file}" # adding a .gz suffix to the compressed dataset file.
        gz_file = f"{compressed_file}.gz"

        with h5py.File(compressed_file, "w") as h5obj_out_comp:
            for name, data in self.dataset:
                # also the chunks flag has been set
                # compression_opts sets the compression depth.
                h5obj_out_comp.create_dataset(name, data=data, compression="gzip", chunks=True, compression_opts=9)

        with open(compressed_file, "rb") as h5file, \
            gzip.open(gz_file , "wb") as gzfile:
                gzfile.writelines(h5file)

        # so you can't directly read the h5 file if it has an outer layer of gzipping.
        with h5py.File(gz_file, "r") as h5obj: 
            name, data = choice(self.dataset)
            assert h5obj.get(name)[:][-1] == data[-1]
    
            


        



    
