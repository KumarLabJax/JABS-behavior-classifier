"""Tests for module-level helpers in feature_base_class."""

import numpy as np
import pytest
from scipy import signal
from scipy.signal import detrend

from jabs.feature_extraction.feature_base_class import _linear_detrend_numpy


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
        """STFT using _linear_detrend_numpy as a callable must equal detrend='linear'."""
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
