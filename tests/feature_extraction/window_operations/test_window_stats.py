"""Unit tests for the sliding window statistics helpers."""

import warnings

import numpy as np
import pytest
from scipy import stats

from jabs.feature_extraction.window_operations import window_stats


def test_pad_sliding_window_shape() -> None:
    """The view has one row per input frame and 2 * window + 1 columns."""
    values = np.arange(5, dtype=np.float64)
    view = window_stats.pad_sliding_window(values, window=2, pad_const=np.nan)
    assert view.shape == (values.size, 2 * 2 + 1)


def test_pad_sliding_window_edge_padding() -> None:
    """A None pad constant repeats the first and last values at the edges."""
    values = np.array([3.0, 4.0, 5.0])
    view = window_stats.pad_sliding_window(values, window=2, pad_const=None)
    np.testing.assert_array_equal(view[0], [3.0, 3.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(view[-1], [3.0, 4.0, 5.0, 5.0, 5.0])


@pytest.mark.parametrize("pad_const", [0.0, -1.0, np.nan], ids=["zero", "negative", "nan"])
def test_pad_sliding_window_uses_constant_padding(pad_const: float) -> None:
    """Any non-None pad constant is used verbatim, including a falsy 0.0."""
    values = np.array([3.0, 4.0, 5.0])
    view = window_stats.pad_sliding_window(values, window=1, pad_const=pad_const)
    np.testing.assert_array_equal(view[0], [pad_const, 3.0, 4.0])
    np.testing.assert_array_equal(view[-1], [4.0, 5.0, pad_const])


def test_get_window_masks_marks_pad_values_invalid() -> None:
    """Values equal to the pad constant are False, real values are True."""
    view = np.array([[np.nan, 1.0, 2.0], [1.0, 2.0, 3.0]])
    masks = window_stats.get_window_masks(view, np.nan)
    np.testing.assert_array_equal(masks, [[False, True, True], [True, True, True]])


def test_get_window_masks_non_nan_constant() -> None:
    """A non-nan pad constant is compared by equality."""
    view = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    masks = window_stats.get_window_masks(view, 0.0)
    np.testing.assert_array_equal(masks, [[False, True, True], [True, True, True]])


def test_get_window_masks_all_pad_row_is_all_valid() -> None:
    """A row consisting entirely of pad values is left fully unmasked."""
    view = np.array([[np.nan, np.nan, np.nan], [1.0, np.nan, 3.0]])
    masks = window_stats.get_window_masks(view, np.nan)
    np.testing.assert_array_equal(masks, [[True, True, True], [True, False, True]])


def test_window_mean_ignores_padding() -> None:
    """The mean at each frame only averages real values within the window."""
    values = np.array([1.0, 2.0, 3.0])
    result = window_stats.window_mean(values, window=1)
    np.testing.assert_allclose(result, [1.5, 2.0, 2.5])


def test_window_median_ignores_padding() -> None:
    """The median at each frame only considers real values within the window."""
    values = np.array([1.0, 2.0, 10.0])
    result = window_stats.window_median(values, window=1)
    np.testing.assert_allclose(result, [1.5, 2.0, 6.0])


def test_window_std_dev_ignores_padding() -> None:
    """The standard deviation at each frame only considers real values."""
    values = np.array([1.0, 3.0, 5.0])
    result = window_stats.window_std_dev(values, window=1)
    np.testing.assert_allclose(result, [1.0, np.std([1.0, 3.0, 5.0]), 1.0])


def test_window_min_and_max() -> None:
    """window_min and window_max reduce over the valid window values."""
    values = np.array([4.0, 1.0, 7.0, 2.0])
    np.testing.assert_allclose(window_stats.window_min(values, window=1), [1.0, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(window_stats.window_max(values, window=1), [4.0, 7.0, 7.0, 7.0])


def test_window_min_and_max_all_nan_input() -> None:
    """An all-nan input yields an all-nan result rather than raising."""
    values = np.full(4, np.nan)
    assert np.all(np.isnan(window_stats.window_min(values, window=1)))
    assert np.all(np.isnan(window_stats.window_max(values, window=1)))


def test_np_skew_matches_window_skew() -> None:
    """window_skew is np_skew applied to the padded sliding window view."""
    values = np.array([1.0, 2.0, 8.0, 3.0, 5.0])
    view = window_stats.pad_sliding_window(values, window=2, pad_const=np.nan)
    np.testing.assert_allclose(
        window_stats.window_skew(values, window=2), window_stats.np_skew(view)
    )


def test_np_kurtosis_matches_window_kurtosis() -> None:
    """window_kurtosis is np_kurtosis applied to the padded sliding window view."""
    values = np.array([1.0, 2.0, 8.0, 3.0, 5.0])
    view = window_stats.pad_sliding_window(values, window=2, pad_const=np.nan)
    np.testing.assert_allclose(
        window_stats.window_kurtosis(values, window=2), window_stats.np_kurtosis(view)
    )


def test_np_skew_matches_scipy_default() -> None:
    """np_skew reproduces scipy's biased skew over the non-nan values of each window."""
    view = np.array([[np.nan, 1.0, 2.0, 8.0], [1.0, 2.0, 8.0, 3.0], [2.0, 8.0, 3.0, np.nan]])
    np.testing.assert_allclose(
        window_stats.np_skew(view), stats.skew(view, axis=1, nan_policy="omit")
    )


def test_np_kurtosis_is_pearson_not_fisher() -> None:
    """np_kurtosis reports Pearson kurtosis, i.e. scipy's ``fisher=False``.

    Guards the documented convention: scipy's default subtracts 3.0, so swapping in
    ``scipy.stats.kurtosis`` with default arguments would silently shift every
    ``kurtosis`` window feature.
    """
    view = np.array([[np.nan, 1.0, 2.0, 8.0], [1.0, 2.0, 8.0, 3.0], [2.0, 8.0, 3.0, np.nan]])
    result = window_stats.np_kurtosis(view)
    np.testing.assert_allclose(
        result, stats.kurtosis(view, axis=1, nan_policy="omit", fisher=False)
    )
    np.testing.assert_allclose(result - 3.0, stats.kurtosis(view, axis=1, nan_policy="omit"))


def _sliding_window_oracle(values: np.ndarray, window: int, op) -> np.ndarray:
    """Apply ``op`` to the non-nan values of each (edge-clipped) window, one frame at a time."""
    result = np.full(values.shape, np.nan)
    for i in range(values.size):
        in_window = values[max(0, i - window) : i + window + 1]
        valid = in_window[~np.isnan(in_window)]
        if valid.size:
            result[i] = op(valid)
    return result


def _window_test_series() -> dict[str, np.ndarray]:
    """Series exercising nan gaps, nan runs wider than the window, constants and all-nan."""
    rng = np.random.default_rng(seed=7)
    n = 120
    scattered_nan = rng.normal(0.0, 20.0, n)
    scattered_nan[rng.random(n) < 0.2] = np.nan
    nan_runs = rng.normal(0.0, 20.0, n)
    nan_runs[:15] = np.nan
    nan_runs[60:80] = np.nan
    mostly_zero = np.zeros(n)
    mostly_zero[40:55] = rng.random(15)
    return {
        "scattered_nan": scattered_nan,
        "nan_runs": nan_runs,
        "constant": np.full(n, 3.0),
        "all_nan": np.full(n, np.nan),
        "mostly_zero": mostly_zero,
    }


@pytest.mark.parametrize("window", [1, 3, 10], ids=["w1", "w3", "w10"])
@pytest.mark.parametrize("case", list(_window_test_series()))
@pytest.mark.parametrize(
    ("op_name", "oracle_op"),
    [
        ("window_median", np.median),
        ("window_min", np.min),
        ("window_max", np.max),
        ("window_skew", stats.skew),
        ("window_kurtosis", lambda x: stats.kurtosis(x, fisher=False)),
    ],
)
def test_window_ops_match_per_window_reference(
    op_name: str, oracle_op, case: str, window: int
) -> None:
    """Each sliding window op agrees with applying the numpy/scipy op to one window at a time."""
    values = _window_test_series()[case]

    with np.errstate(all="ignore"):
        result = getattr(window_stats, op_name)(values, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = _sliding_window_oracle(values, window, oracle_op)

    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9, equal_nan=True)
