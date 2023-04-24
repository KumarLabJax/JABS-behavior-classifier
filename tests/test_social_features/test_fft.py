import numpy as np
import unittest

# 1. Sanity checks


class TestScientificComputingBasics(unittest.TestCase):
    pass

# 2. Signal Processing
class TestRollingWindow(unittest.TestCase):
    '''
    This test is designed for the utils/utilities.py rolling_window
    method.  This is used in the Feature base class object to compute
    window features.
    '''

    log = True 

    @classmethod
    def setUpClass(cls) -> None:
        import src.utils.utilities as utils
        cls.rolling_window = utils.rolling_window

    @unittest.skip("resolved")
    def test_rolling_window(self):
        '''
        segmentation data has the following format:
        (frames, identities, contours, contour length, xy-point)

        Spent too much time here.  
        Basically I am curious about the difference in invoking feature_base_class/_compute_window_feature/rolling_window
        on 1D vs 2D features.  It is not obvious to me that the 2D rolling is correct.

        1D: (frames - window_size, window_size) vs 2D: (frames, features - window_size, window_size) 
        '''
        step_size = 1
        window_size = 6
        num_frames = 3600

        data = [
            ('points', (num_frames, 5, 12, 2)),       # ignore
            ('seg data', (num_frames, 5, 4, 319, 2)), # ignore
            ('1D', (num_frames, )),
            ('2D', (num_frames, 8))
        ]

        # The way this is used in code is for feature_values arrays.  Let me check the shape of these arrays.
        # It appears to be (number of frames,), which makes intuitive sense for 1D arrays, but what about 2D 
        # feature values arrays such as hu_moments, moments, and ellipse_fitting which all have shapes of the
        # form: (num_frames, len(self._feature_names)).  I checked and pairwise_distances also uses this 
        # 2D structure.  Next I will check if pairwise_distances is ever used with rolling_window. 

        # resolved:
        # I overlooked something obvious, 
        # The rolling_window code is used with: if feature_values.ndim == 1: 
        # thus it is not applicable for 2D features like I was worried about.
        
        rolling_shape = lambda shape: shape[:-1] + (shape[-1] - window_size + 1 - step_size + 1, window_size)
        
        for name, shape in data[2:]:

            A = np.random.randint(0, 1000, shape)
            roll_shape = rolling_shape(shape)
  
            window = TestRollingWindow.rolling_window(A, window_size)

            if TestRollingWindow.log:
                print(f"{name} shape: ", shape, "roll shape:", roll_shape)

            assert len(roll_shape) - 1 == len(shape)
            assert window.shape == roll_shape
            assert isinstance(window, np.ndarray)


