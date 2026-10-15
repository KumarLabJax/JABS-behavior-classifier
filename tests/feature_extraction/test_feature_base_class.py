"""Tests for module-level helpers in feature_base_class."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal, stats
from scipy.signal import detrend

from jabs.feature_extraction.feature_base_class import Feature, _linear_detrend_numpy


class TestLinearDetrendNumpy:
    """Verify that _linear_detrend_numpy matches scipy.signal.detrend(..., type='linear').

    scipy.signal.stft calls the detrend callable with a 2-D array of shape
    (n_windows, nperseg), expecting each row to be detrended independently.
    """

    def _scipy_detrend_rows(self, x: np.ndarray) -> np.ndarray:
        """Apply scipy linear detrend along the last axis (row-wise)."""
        return detrend(x, axis=-1, type="linear")

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_pure_linear_signal_becomes_zero(self) -> None:
        """A signal that is already a pure linear trend should detrend to zero."""
        t = np.linspace(0, 1, 50)
        x = 3.0 * t + 7.0  # y = 3t + 7
        result = _linear_detrend_numpy(x[np.newaxis, :])  # shape (1, 50)
        assert result == pytest.approx(np.zeros((1, 50)), abs=1e-10)

    def test_constant_signal_becomes_zero(self) -> None:
        """A constant signal has a trivial linear trend and detrends to zero."""
        x = np.full((4, 30), 5.0)
        result = _linear_detrend_numpy(x)
        assert result == pytest.approx(np.zeros_like(x), abs=1e-10)

    def test_already_zero_mean_no_trend(self) -> None:
        """A signal with no trend should be unchanged (up to floating-point)."""
        # Construct signal with exactly zero slope by making it symmetric
        x = np.sin(np.linspace(0, 2 * np.pi, 64))
        x2d = x[np.newaxis, :]
        result = _linear_detrend_numpy(x2d)
        expected = self._scipy_detrend_rows(x2d)
        assert result == pytest.approx(expected, abs=1e-12)

    # ------------------------------------------------------------------
    # Matches scipy row-wise detrend for 2-D inputs
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n_windows,nperseg", [(1, 21), (8, 21), (16, 41), (32, 11)])
    def test_matches_scipy_random_2d(self, n_windows: int, nperseg: int) -> None:
        """Output must match scipy detrend for random 2-D arrays of various shapes."""
        rng = np.random.default_rng(seed=42)
        x = rng.standard_normal((n_windows, nperseg))
        result = _linear_detrend_numpy(x)
        expected = self._scipy_detrend_rows(x)
        assert result == pytest.approx(expected, rel=1e-10, abs=1e-12)

    def test_matches_scipy_sinusoid_plus_linear_trend(self) -> None:
        """Sinusoid contaminated with a linear trend detrends to the clean sinusoid."""
        t = np.linspace(0, 1, 51)
        clean = np.sin(2 * np.pi * 3 * t)
        trend = 2.0 * t - 1.0
        x = (clean + trend)[np.newaxis, :]

        result = _linear_detrend_numpy(x)
        expected = self._scipy_detrend_rows(x)
        assert result == pytest.approx(expected, rel=1e-10, abs=1e-12)

    def test_matches_scipy_multi_row_mixed_trends(self) -> None:
        """Each row can have a different linear trend; all should be removed correctly."""
        n = 40
        t = np.linspace(0, 1, n)
        rows = np.vstack(
            [
                1.0 * t + np.sin(2 * np.pi * t),  # positive slope
                -5.0 * t + np.cos(4 * np.pi * t),  # negative slope
                np.zeros(n),  # flat
                10.0 * t,  # pure trend
            ]
        )
        result = _linear_detrend_numpy(rows)
        expected = self._scipy_detrend_rows(rows)
        assert result == pytest.approx(expected, rel=1e-10, abs=1e-12)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_single_sample_returns_zeros(self) -> None:
        """A 1-sample segment should return zeros (matches scipy behavior, avoids divide-by-zero)."""
        x = np.array([[3.0], [7.0], [-1.5]])
        result = _linear_detrend_numpy(x)
        assert result == pytest.approx(np.zeros_like(x), abs=1e-15)

    def test_single_sample_matches_scipy(self) -> None:
        """Single-sample output must match scipy.signal.detrend."""
        x = np.array([[5.0]])
        result = _linear_detrend_numpy(x)
        expected = self._scipy_detrend_rows(x)
        assert result == pytest.approx(expected, abs=1e-14)

    # ------------------------------------------------------------------
    # End-to-end: STFT output is identical with callable vs detrend="linear"
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("fps", [30, 60])
    def test_stft_output_matches_linear_detrend(self, fps: int) -> None:
        """STFT magnitude using _linear_detrend_numpy must match detrend='linear'.

        Only magnitude equivalence is required: downstream PSD calculations
        depend on |zxx|, not on complex phase.
        """
        rng = np.random.default_rng(seed=7)
        window_size = 10
        n_frames = 200
        x = rng.standard_normal(n_frames)

        common_kwargs = {
            "fs": fps,
            "nperseg": window_size * 2 + 1,
            "noverlap": window_size * 2,
            "window": "hann",
            "scaling": "psd",
        }

        _, _, zxx_scipy = signal.stft(x, detrend="linear", **common_kwargs)
        _, _, zxx_numpy = signal.stft(x, detrend=_linear_detrend_numpy, **common_kwargs)

        assert np.abs(zxx_numpy) == pytest.approx(np.abs(zxx_scipy), rel=1e-9, abs=1e-14)


def _reference_circular(
    values: np.ndarray, mask: np.ndarray, window_size: int, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame scipy circmean/circstd, the implementation the vectorized path replaced."""
    n = len(values)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(n):
        if not mask[i]:
            continue
        start = max(0, i - window_size)
        window = values[start : i + window_size + 1][mask[start : i + window_size + 1]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean[i] = stats.circmean(window, low=low, high=high, nan_policy="omit")
            std[i] = stats.circstd(window, low=low, high=high, nan_policy="omit")
    return mean, std


@pytest.mark.parametrize(
    ("low", "high"), [(-180.0, 180.0), (0.0, 360.0)], ids=["bearing", "zero_to_360"]
)
@pytest.mark.parametrize("window_size", [1, 5, 40], ids=["w1", "w5", "w40"])
def test_circular_window_features_match_scipy(low: float, high: float, window_size: int) -> None:
    """The vectorized circular window features agree with per-window scipy calls."""
    rng = np.random.default_rng(seed=0)
    n = 300
    values = rng.uniform(low, high, n)
    values[rng.random(n) < 0.15] = np.nan
    values[100:120] = np.nan  # a run of nan wider than some windows
    mask = rng.random(n) > 0.1
    mask[200:210] = False  # identity absent

    feature = SimpleNamespace(_circular_range=(low, high))
    result = Feature._compute_window_features_circular(
        feature, {"angle": values}, mask, window_size
    )

    expected_mean, expected_std = _reference_circular(values, mask, window_size, low, high)
    np.testing.assert_allclose(result["mean"]["angle"], expected_mean, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(result["std_dev"]["angle"], expected_std, atol=1e-4, equal_nan=True)


def test_circular_window_features_edge_cases() -> None:
    """All-nan windows are nan, constant windows have zero std, and the wrap is handled."""
    feature = SimpleNamespace(_circular_range=(-180.0, 180.0))
    mask = np.ones(6, dtype=bool)
    values = np.array([np.nan, np.nan, 10.0, 10.0, 170.0, -170.0])

    result = Feature._compute_window_features_circular(feature, {"a": values}, mask, 0)

    assert np.isnan(result["mean"]["a"][:2]).all()
    assert np.isnan(result["std_dev"]["a"][:2]).all()
    assert result["std_dev"]["a"][2] == pytest.approx(0.0, abs=1e-6)

    result = Feature._compute_window_features_circular(feature, {"a": values}, mask, 1)
    # window around the last frame holds 170 and -170, whose circular mean is +-180
    assert abs(result["mean"]["a"][5]) == pytest.approx(180.0)
