
## 4/12/2023

+ Reading
    + numpy strides
    + scipy signals
    + sleep paper
+ Other
    + Rereading <a href="https://github.com/KumarLabJax/MouseSleep/blob/main/dataset-creation/Sleep_feature_generation.py#L33">sleep code</a>
    
    + <a href="https://jacksonlaboratory.atlassian.net/l/cp/cvvtKzAY  ">Design plan confluence page</a>
    + Trying to understand if rolling window method is valid for 2D features.

## 4/18/2023
+ Reading
    + Fourier series

## 4/19/2023
+ Reading
    + numpy masked arrays
    + spectral density
    + Welch's method


## 4/24/2023
+ reviewing code
    + broken setup_windows.bat
        fix:
            python -m venv jabs.venv
            jabs.venv\Scripts\activate
            pip install -r requirements.txt
        + look into docker container for windows test.
    + broken shapely module referenced in requirements.txt
        fix:
            pip uninstall shapely; pip install shapely
+ what is typing -> type checking?
    addresses circular imports
+ compute perimeter of a given mouse from the contour data.
+ auto-generate requirements.txt
    pip freeze > requirements2.txt
+ finished initial version of fft features.

# 4/25/2023
+ review  <a href="https://jacksonlaboratory.atlassian.net/l/cp/yse7N20P">notes</a> from Brian meeting & plan fft re-write.
+ demo
+ started to re-write fft
+ read about numpy masks & masked operations


# 5/9

+ need to rewrite signal processing.  in the signal processing method currently, sub_op returns a dictionary and and I am
trying to stick this into a numpy array which is not valid.  I need to do the opposite, load a dictionary with numpy arrays.   
