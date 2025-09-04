import warnings

import numpy as np


def pad_sliding_window(arr: np.ndarray, window: int, pad_const: float | None = None) -> np.ndarray:
    """Generates a sliding window view of an input array with nan-padding.

    Args:
        arr: 1d array to generate a sliding window
        window: half size of sliding window
        pad_const: Constant to pad the array with. None will pad with value at the edge.

    Returns:
        an unmodifiable 2d view of the input array where the first axis is time and the second axis is the window. Note that typical usage will use summary stats with axis=1.
    """
    if pad_const:
        arr_ext = np.concatenate([np.full(window, pad_const), arr, np.full(window, pad_const)])
    else:
        arr_ext = np.concatenate([np.full(window, arr[0]), arr, np.full(window, arr[-1])])
    return np.lib.stride_tricks.sliding_window_view(arr_ext, window_shape=window * 2 + 1)


def get_window_masks(sliding_window_view: np.ndarray, const: float) -> np.ndarray:
    """Creates a mask for invalid values in a sliding window matrix.

    Args:
        sliding_window_view: sliding window matrix from `pad_sliding_window`
        const: constant pad value

    Returns:
        matrix describing valid (0) and invalid (1) window values
    """
    if np.isnan(const):
        window_masks = ~np.isnan(sliding_window_view)
    else:
        window_masks = sliding_window_view != const
    for no_data_row in np.where(np.all(~window_masks, axis=1)):
        window_masks[no_data_row] = True

    return window_masks


def window_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked mean of a window

    Args:
        values: 1d np.ndarray of values
        window: window size

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
        window: window size

    Returns:
        sliding window median values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    window_masks = get_window_masks(window_values, np.nan)
    return np.ma.median(np.ma.array(window_values, mask=~window_masks), axis=1).filled()


def window_std_dev(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked standard deviation of a window

    Args:
        values: 1d np.ndarray of values
        window: window size

    Returns:
        sliding window standard deviation values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return_values = np.nanstd(window_values, axis=1)
    return return_values


def np_kurtosis(values: np.ndarray) -> np.ndarray:
    """Calculates kurtosis in a rolling window faster than scipy

    Args:
        values: 2d array of with time and a window

    Returns:
        kurtosis values that match scipy.stats.kurtosis(values, axis, nan_policy='omit')

    Raises:
        RuntimeWarning when an entire window is nans
    """
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1)
    counts = np.sum(~np.isnan(values), axis=1)
    return np.nansum((values - np.tile(mean, [values.shape[1], 1]).T) ** 4, axis=1) / (
        counts * std**4
    )


def window_kurtosis(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked kurtosis of a window

    Args:
        values: 1d np.ndarray of values
        window: window size

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

    Args:
        values: 2d array of with time and a window

    Returns:
        skew values that match scipy.stats.skew(values, axis, nan_policy='omit')

    Raises:
        RuntimeWarning when an entire window is nans
    """
    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1)
    counts = np.sum(~np.isnan(values), axis=1)
    return np.nansum((values - np.tile(mean, [values.shape[1], 1]).T) ** 3, axis=1) / (
        counts * std**3
    )


def window_skew(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked skew of a window

    Args:
        values: 1d np.ndarray of values
        window: window size

    Returns:
        sliding window skew values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return_values = np_skew(window_values)
    return return_values


def window_min(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked maximum of a window

    Args:
        values: 1d np.ndarray of values
        window: window size

    Returns:
        sliding window maximum values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    window_masks = get_window_masks(window_values, np.nan)
    if np.all(np.isnan(values)):
        return np.full(values.shape, np.nan)
    return np.min(window_values, axis=1, initial=np.nanmax(values), where=window_masks)


def window_max(values: np.ndarray, window: int) -> np.ndarray:
    """Calculates a masked maximum of a window

    Args:
        values: 1d np.ndarray of values
        window: window size

    Returns:
        sliding window maximum values
    """
    window_values = pad_sliding_window(values, window, pad_const=np.nan)
    window_masks = get_window_masks(window_values, np.nan)
    if np.all(np.isnan(values)):
        return np.full(values.shape, np.nan)
    return np.max(window_values, axis=1, initial=np.nanmin(values), where=window_masks)
