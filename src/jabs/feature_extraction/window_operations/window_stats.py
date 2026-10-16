import warnings

import numpy as np


def pad_sliding_window(arr: np.ndarray, window: int, pad_const: float | None = None) -> np.ndarray:
    """Generates a sliding window view of an input array with nan-padding.

    Args:
        arr: 1d array to generate a sliding window
        window: number of frames to include on each side of the center frame. The
            resulting window spans ``2 * window + 1`` frames.
        pad_const: Constant to pad the array with. None will pad with value at the edge.

    Returns:
        an unmodifiable 2d view of the input array where the first axis is time and the second axis is the window. Note that typical usage will use summary stats with axis=1.
    """
    if pad_const is not None:
        arr_ext = np.concatenate([np.full(window, pad_const), arr, np.full(window, pad_const)])
    else:
        arr_ext = np.concatenate([np.full(window, arr[0]), arr, np.full(window, arr[-1])])
    return np.lib.stride_tricks.sliding_window_view(arr_ext, window_shape=window * 2 + 1)


def window_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked mean of a window

    Args:
        values: 1d np.ndarray of values
        window: number of frames to include on each side of the center frame. The
            window used for each output value spans ``2 * window + 1`` frames.

    Returns:
        sliding window mean values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return_values = np.nanmean(window_values, axis=1)
    return return_values


def window_median(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked median of a window

    Args:
        values: 1d np.ndarray of values
        window: number of frames to include on each side of the center frame. The
            window used for each output value spans ``2 * window + 1`` frames.

    Returns:
        sliding window median values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    # nan sorts to the end of each row, so the first `counts` entries are the valid values
    sorted_values = np.sort(window_values, axis=1)
    counts = np.sum(~np.isnan(window_values), axis=1)
    lower = np.take_along_axis(sorted_values, np.maximum((counts - 1) // 2, 0)[:, None], axis=1)
    upper = np.take_along_axis(sorted_values, (counts // 2)[:, None], axis=1)
    return np.where(counts > 0, (lower[:, 0] + upper[:, 0]) / 2, np.nan)


def window_std_dev(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked standard deviation of a window

    Args:
        values: 1d np.ndarray of values
        window: number of frames to include on each side of the center frame. The
            window used for each output value spans ``2 * window + 1`` frames.

    Returns:
        sliding window standard deviation values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return_values = np.nanstd(window_values, axis=1)
    return return_values


def _center_windows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtracts each window's mean from its non-nan values.

    Args:
        values: 2d array with time on the first axis and the window on the second

    Returns:
        tuple of the number of non-nan values in each window and the array of window
        values minus their window mean (nan stays nan). A window that is entirely nan
        has a count of 0 and a nan mean.
    """
    valid = ~np.isnan(values)
    counts = np.sum(valid, axis=1)
    mean = np.where(valid, values, 0.0).sum(axis=1) / counts
    return counts, values - mean[:, None]


def np_kurtosis(values: np.ndarray) -> np.ndarray:
    """Calculates kurtosis in a rolling window faster than scipy

    This computes Pearson (non-excess) kurtosis, equivalent to
    ``scipy.stats.kurtosis(values, axis=1, nan_policy="omit", fisher=False)``. It is
    *not* equivalent to scipy's default, which subtracts 3.0 to give Fisher (excess)
    kurtosis: values returned here are 3.0 larger than
    ``scipy.stats.kurtosis(values, axis=1, nan_policy="omit")``.

    Note that ``signal_stats.psd_kurtosis`` calls scipy with its default arguments, so
    the ``kurtosis`` window feature and the ``psd_kurtosis`` signal feature do not use
    the same convention.

    Args:
        values: 2d array with time on the first axis and the window on the second

    Returns:
        Pearson kurtosis of the non-nan values in each window. A window that is
        entirely nan yields nan.

    Note:
        numpy emits (does not raise) a RuntimeWarning when an entire window is nan.
        Callers such as `window_kurtosis` suppress that warning.
    """
    counts, centered = _center_windows(values)
    squared = centered * centered
    variance = np.nansum(squared, axis=1) / counts
    return np.nansum(squared * squared, axis=1) / counts / (variance * variance)


def window_kurtosis(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked kurtosis of a window

    Args:
        values: 1d np.ndarray of values
        window: number of frames to include on each side of the center frame. The
            window used for each output value spans ``2 * window + 1`` frames.

    Returns:
        sliding window kurtosis values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return_values = np_kurtosis(window_values)
    return return_values


def np_skew(values: np.ndarray) -> np.ndarray:
    """Calculates skew in a rolling window faster than scipy

    This computes the biased (population) skew, equivalent to
    ``scipy.stats.skew(values, axis=1, nan_policy="omit")``, which is scipy's default.

    Args:
        values: 2d array with time on the first axis and the window on the second

    Returns:
        skew of the non-nan values in each window. A window that is entirely nan
        yields nan.

    Note:
        numpy emits (does not raise) a RuntimeWarning when an entire window is nan.
        Callers such as `window_skew` suppress that warning.
    """
    counts, centered = _center_windows(values)
    squared = centered * centered
    variance = np.nansum(squared, axis=1) / counts
    return np.nansum(squared * centered, axis=1) / counts / variance**1.5


def window_skew(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked skew of a window

    Args:
        values: 1d np.ndarray of values
        window: number of frames to include on each side of the center frame. The
            window used for each output value spans ``2 * window + 1`` frames.

    Returns:
        sliding window skew values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return_values = np_skew(window_values)
    return return_values


def window_min(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked minimum of a window

    Args:
        values: 1d np.ndarray of values
        window: number of frames to include on each side of the center frame. The
            window used for each output value spans ``2 * window + 1`` frames.

    Returns:
        sliding window minimum values
    """
    # fmin ignores nan, and is nan only when every value in the window is nan
    return np.fmin.reduce(pad_sliding_window(values, window, pad_const=np.nan), axis=1)


def window_max(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked maximum of a window

    Args:
        values: 1d np.ndarray of values
        window: number of frames to include on each side of the center frame. The
            window used for each output value spans ``2 * window + 1`` frames.

    Returns:
        sliding window maximum values
    """
    # fmax ignores nan, and is nan only when every value in the window is nan
    return np.fmax.reduce(pad_sliding_window(values, window, pad_const=np.nan), axis=1)
