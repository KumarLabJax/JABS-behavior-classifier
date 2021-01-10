import os
import sys
from contextlib import contextmanager

import numpy as np


@contextmanager
def hide_stderr():
    fd = sys.stderr.fileno()

    # copy fd before it is overwritten
    with os.fdopen(os.dup(fd), 'wb') as copied:
        sys.stderr.flush()

        # open destination
        with open(os.devnull, 'wb') as fout:
            os.dup2(fout.fileno(), fd)
        try:
            yield fd
        finally:
            # restore stderr to its previous value
            sys.stderr.flush()
            os.dup2(copied.fileno(), fd)


def rolling_window(a, window, step_size=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def smooth(vec, smoothing_window):
    if smoothing_window <= 1 or len(vec) == 0:
        return vec.astype(np.float)
    else:
        assert smoothing_window % 2 == 1, 'expected smoothing_window to be odd'
        half_conv_len = smoothing_window // 2
        smooth_tgt = np.concatenate([
            np.full(half_conv_len, vec[0], dtype=vec.dtype),
            vec,
            np.full(half_conv_len, vec[-1], dtype=vec.dtype),
        ])

        smoothing_val = 1 / smoothing_window
        conv_arr = np.full(smoothing_window, smoothing_val)

        return np.convolve(smooth_tgt, conv_arr, mode='valid')
