"""Unit tests for the power spectral density summary statistics."""

import numpy as np
import pytest
from scipy import stats

from jabs.feature_extraction.window_operations import signal_stats

# scipy warns about the degenerate columns and all-nan slices these tests use on purpose
pytestmark = [
    pytest.mark.filterwarnings("ignore:Precision loss:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:After omitting NaNs"),
]


def _psd_matrices() -> dict[str, np.ndarray]:
    """PSD-like matrices (frequencies on the first axis, frames on the second)."""
    rng = np.random.default_rng(seed=11)
    random_power = np.abs(rng.standard_normal((21, 80))) ** 2

    with_flat_columns = random_power.copy()
    with_flat_columns[:, 10:20] = 0.0  # identity absent: all-zero spectra
    with_flat_columns[:, 30] = 1.0  # constant, non-zero spectrum
    with_flat_columns[:, 31] = 1e-30  # tiny but constant

    wide_dynamic_range = random_power * np.logspace(-20, 10, 80)

    return {
        "random": random_power,
        "zero_and_constant_columns": with_flat_columns,
        "wide_dynamic_range": wide_dynamic_range,
        "two_frequencies": random_power[:2],
    }


@pytest.mark.parametrize("case", list(_psd_matrices()))
def test_psd_skew_matches_scipy(case: str) -> None:
    """psd_skew equals scipy's biased skew per column, with non-finite results as nan."""
    psd = _psd_matrices()[case]

    expected = np.nan_to_num(stats.skew(psd, axis=0), nan=np.nan, posinf=np.nan, neginf=np.nan)

    np.testing.assert_allclose(
        signal_stats.psd_skew(None, psd), expected, rtol=1e-9, atol=1e-12, equal_nan=True
    )


@pytest.mark.parametrize("case", list(_psd_matrices()))
def test_psd_kurtosis_matches_scipy(case: str) -> None:
    """psd_kurtosis equals scipy's Fisher kurtosis per column, with non-finite results as nan."""
    psd = _psd_matrices()[case]

    expected = np.nan_to_num(stats.kurtosis(psd, axis=0), nan=np.nan, posinf=np.nan, neginf=np.nan)

    np.testing.assert_allclose(
        signal_stats.psd_kurtosis(None, psd), expected, rtol=1e-9, atol=1e-12, equal_nan=True
    )


def test_psd_skew_and_kurtosis_of_all_zero_columns_are_nan() -> None:
    """A spectrum with no power has no defined shape."""
    psd = np.zeros((21, 4))

    assert np.isnan(signal_stats.psd_skew(None, psd)).all()
    assert np.isnan(signal_stats.psd_kurtosis(None, psd)).all()


def test_psd_skew_and_kurtosis_omit_nan_values() -> None:
    """Input containing nan falls back to scipy's nan-omitting statistics."""
    rng = np.random.default_rng(seed=5)
    psd = np.abs(rng.standard_normal((21, 6))) ** 2
    psd[3, 1] = np.nan
    psd[:, 4] = np.nan

    skew = signal_stats.psd_skew(None, psd)
    kurtosis = signal_stats.psd_kurtosis(None, psd)

    np.testing.assert_allclose(skew, stats.skew(psd, axis=0, nan_policy="omit"), equal_nan=True)
    np.testing.assert_allclose(
        kurtosis, stats.kurtosis(psd, axis=0, nan_policy="omit"), equal_nan=True
    )
    assert np.isfinite(skew[1])
    assert np.isnan(skew[4])
