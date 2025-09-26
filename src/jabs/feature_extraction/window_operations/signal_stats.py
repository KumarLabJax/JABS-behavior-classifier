import warnings

import numpy as np
from scipy.stats import kurtosis, skew


def psd_sum(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the sum power spectral density

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix

    Returns:
        sum of power
    """
    return np.sum(psd, axis=0)


def psd_max(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the max power

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix

    Returns:
        max of power
    """
    return np.nanmax(psd, axis=0)


def psd_min(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the min power

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix

    Returns:
        min of power
    """
    return np.min(psd, axis=0)


def psd_mean(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the mean power spectral density

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix

    Returns:
        mean of power
    """
    return np.mean(psd, axis=0)


def psd_mean_band(
    freqs: np.ndarray,
    psd: np.ndarray,
    band_low: int = 0,
    band_high: float = np.finfo(np.float64).max,
) -> np.ndarray:
    """Calculates the mean power spectral density in a band

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix
        band_low: lower bound of the frequency band
        band_high: upper bound of the frequency band

    Returns:
        mean of power
    """
    idx = np.logical_and(freqs >= band_low, freqs < band_high)

    if not np.any(idx):
        return np.full([psd.shape[1]], np.nan)
    return np.mean(np.asarray(psd)[idx], axis=0)


def psd_median(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the median power spectral density

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix

    Returns:
        median of power
    """
    return np.median(psd, axis=0)


def psd_std_dev(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the standard deviation power spectral density

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix

    Returns:
        standard deviation of power
    """
    return np.std(psd, axis=0)


def psd_kurtosis(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the kurtosis power spectral density

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix

    Returns:
        kurtosis of power
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return_values = kurtosis(psd, axis=0, nan_policy="omit")
        # If infinity shows up, convert to nan
        return_values = np.nan_to_num(return_values, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return return_values


def psd_skew(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the skew power spectral density

    Args:
        freqs: frequencies in the psd, ignored
        psd: power spectral density matrix

    Returns:
        skew of power
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return_values = skew(psd, axis=0, nan_policy="omit")
        # If infinity shows up, convert to nan
        return_values = np.nan_to_num(return_values, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return return_values


def psd_peak_freq(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Calculates the frequency with the most power

    Args:
        freqs: frequencies in the psd
        psd: power spectral density matrix

    Returns:
        frequency with highest power
    """
    return freqs[np.argmax(psd, axis=0)]
