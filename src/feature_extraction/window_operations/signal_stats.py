from scipy.stats import kurtosis, skew
import numpy as np


def psd_sum(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the sum power spectral density

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: sum of power
    """
    return np.sum(psd, axis=0)

def psd_max(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the max power

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: max of power
    """
    return np.max(psd, axis=0)

def psd_min(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the min power

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: min of power
    """
    return np.min(psd, axis=0)

def psd_mean(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the mean power spectral density

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: mean of power
    """
    return np.mean(psd, axis=0)

def psd_mean_band(freqs: np.ndarray, psd: np.ndarray, band_low: int = 0, band_high: float = np.finfo(np.float64).max) -> np.ndarray:
    """
    Calculates the mean power spectral density in a band

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: mean of power
    """
    idx = np.logical_and(freqs >= band_low, freqs < band_high)
    return np.mean(psd[idx], axis=0)

def psd_median(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the median power spectral density

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: median of power
    """
    return np.median(psd, axis=0)

def psd_std_dev(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the standard deviation power spectral density

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: standard deviation of power
    """
    return np.std_dev(psd, axis=0)

def psd_kurtosis(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the kurtosis power spectral density

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: kurtosis of power
    """
    return kurtosis(psd, axis=0, nan_policy='omit')

def psd_skew(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the skew power spectral density

    :param freqs: frequencies in the psd, ignored
    :param psd: power spectral density matrix
    :return: skew of power
    """
    return skew(psd, axis=0, nan_policy='omit')

def psd_peak_freq(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Calculates the frequency with the most power

    :param freqs: frequencies in the psd
    :param psd: power spectral density matrix
    :return: frequency with highest power
    """
    return freqs[np.argmax(psd, axis=0)]
