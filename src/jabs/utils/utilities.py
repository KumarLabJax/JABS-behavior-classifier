import hashlib
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def hide_stderr() -> int:
    """Context manager to temporarily suppress output to standard error (stderr).

    Redirects all output sent to stderr to os.devnull while the context is active,
    restoring stderr to its original state upon exit.

    Yields:
        int: The file descriptor for stderr.
    """
    fd = sys.stderr.fileno()

    # copy fd before it is overwritten
    with os.fdopen(os.dup(fd), "wb") as copied:
        sys.stderr.flush()

        # open destination
        with open(os.devnull, "wb") as fout:
            os.dup2(fout.fileno(), fd)
        try:
            yield fd
        finally:
            # restore stderr to its previous value
            sys.stderr.flush()
            os.dup2(copied.fileno(), fd)


def rolling_window(a, window, step_size=1):
    """Creates a rolling window view of a 1D numpy array.

    Generates a view of the input array with overlapping windows of the specified size,
    optionally with a custom step size between windows.

    Args:
        a (np.ndarray): Input 1D array.
        window (int): Size of each rolling window.
        step_size (int, optional): Step size between windows. Defaults to 1.

    Returns:
        np.ndarray: A 2D array where each row is a windowed view of the input.

    Raises:
        ValueError: If the window size is larger than the input array.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    strides = (*a.strides, a.strides[-1] * step_size)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def smooth(vec, smoothing_window):
    """Smooths a 1D numpy array using a moving average with edge padding.

    Pads the input vector at both ends with its edge values, then applies a moving average
    of the specified window size. The window size must be odd.

    Args:
        vec (np.ndarray): Input 1D array to smooth.
        smoothing_window (int): Size of the moving average window (must be odd).

    Returns:
        np.ndarray: Smoothed array as float values.

    Raises:
        AssertionError: If `smoothing_window` is not odd.
    """
    if smoothing_window <= 1 or len(vec) == 0:
        return vec.astype(np.float)
    else:
        assert smoothing_window % 2 == 1, "expected smoothing_window to be odd"
        half_conv_len = smoothing_window // 2
        smooth_tgt = np.concatenate(
            [
                np.full(half_conv_len, vec[0], dtype=vec.dtype),
                vec,
                np.full(half_conv_len, vec[-1], dtype=vec.dtype),
            ]
        )

        smoothing_val = 1 / smoothing_window
        conv_arr = np.full(smoothing_window, smoothing_val)

        return np.convolve(smooth_tgt, conv_arr, mode="valid")


def n_choose_r(n, r):
    """compute number of unique selections (disregarding order) of r items from a set of n items

    Args:
        n: number of elements to select from
        r: number of elements to select

    Returns:
        total number of combinations disregarding order
    """
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))


def hash_file(file: Path):
    """return hash"""
    chunk_size = 8192
    with file.open("rb") as f:
        h = hashlib.blake2b(digest_size=20)
        c = f.read(chunk_size)
        while c:
            h.update(c)
            c = f.read(chunk_size)
    return h.hexdigest()


def get_bool_env_var(var_name, default_value=False) -> bool:
    """Gets a boolean value from an environment variable.

    Args:
        var_name: The name of the environment variable.
        default_value: The default value to return if the variable is
            not set or invalid.

    Returns:
        A boolean value.
    """
    value = os.getenv(var_name)
    if value is None:
        return default_value

    return value.lower() in ("true", "1", "yes", "on", "y", "t")
