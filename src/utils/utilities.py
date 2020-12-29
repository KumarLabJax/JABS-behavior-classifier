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
