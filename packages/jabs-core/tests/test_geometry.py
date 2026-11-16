"""Unit tests for jabs.core.utils.geometry module."""

import numpy as np
import pytest

from jabs.core.utils import signed_angle_degrees, wrap_degrees


@pytest.mark.parametrize(
    ("a", "vertex", "c", "expected"),
    [
        ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), 90.0),
        ((1.0, 0.0), (0.0, 0.0), (0.0, -1.0), -90.0),
        ((1.0, 0.0), (0.0, 0.0), (1.0, 0.0), 0.0),
        ((1.0, 0.0), (0.0, 0.0), (-1.0, 0.0), -180.0),
        ((2.0, 1.0), (1.0, 1.0), (1.0, 2.0), 90.0),
    ],
    ids=[
        "quarter-turn-ccw",
        "quarter-turn-cw",
        "collinear-same-side",
        "opposite",
        "offset-vertex",
    ],
)
def test_signed_angle_degrees_single_points(
    a: tuple[float, float],
    vertex: tuple[float, float],
    c: tuple[float, float],
    expected: float,
) -> None:
    """Known three-point configurations produce the expected signed angle."""
    assert signed_angle_degrees(a, vertex, c) == pytest.approx(expected)


def test_signed_angle_degrees_arrays() -> None:
    """An (n, 2) array of points yields one angle per row."""
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    vertex = np.array([[0.0, 0.0], [0.0, 0.0]])
    c = np.array([[0.0, 1.0], [0.0, -1.0]])

    angles = signed_angle_degrees(a, vertex, c)

    assert angles.shape == (2,)
    np.testing.assert_allclose(angles, [90.0, -90.0])


def test_signed_angle_degrees_wraps_to_half_open_range() -> None:
    """Angles are wrapped to [-180, 180), so a half turn reports -180 rather than 180."""
    rng = np.random.default_rng(seed=42)
    a = rng.uniform(-100.0, 100.0, size=(500, 2))
    vertex = rng.uniform(-100.0, 100.0, size=(500, 2))
    c = rng.uniform(-100.0, 100.0, size=(500, 2))

    angles = signed_angle_degrees(a, vertex, c)

    assert np.all(angles >= -180.0)
    assert np.all(angles < 180.0)
    assert signed_angle_degrees((1.0, 0.0), (0.0, 0.0), (-1.0, 0.0)) == pytest.approx(-180.0)


def test_signed_angle_degrees_preserves_float32() -> None:
    """float32 coordinates produce float32 angles, matching the cached feature dtype."""
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    vertex = np.array([[0.0, 0.0]], dtype=np.float32)
    c = np.array([[0.0, 1.0]], dtype=np.float32)

    assert signed_angle_degrees(a, vertex, c).dtype == np.float32


def test_signed_angle_degrees_nan_coordinates() -> None:
    """A missing coordinate propagates as nan rather than raising."""
    angles = signed_angle_degrees(
        np.array([[np.nan, np.nan], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [0.0, 1.0]]),
    )

    assert np.isnan(angles[0])
    assert angles[1] == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (179.0, 179.0),
        (180.0, -180.0),
        (-180.0, -180.0),
        (270.0, -90.0),
        (-190.0, 170.0),
        (720.0, 0.0),
    ],
    ids=["zero", "inside", "plus-180", "minus-180", "above-range", "below-range", "multi-turn"],
)
def test_wrap_degrees(angle: float, expected: float) -> None:
    """Angles are wrapped into [-180, 180), with +180 mapped to -180."""
    assert wrap_degrees(angle) == pytest.approx(expected)


def test_wrap_degrees_float_noise_below_minus_180_stays_in_range() -> None:
    """Float noise just below -180 wraps to -180, not +180.

    ``(angle + 180) % 360`` rounds a tiny negative number up to exactly 360, which the
    plain modulo formula would report as +180.
    """
    angles = np.array([-180.0 - 1e-14, -180.0 - 1e-15])

    wrapped = wrap_degrees(angles)

    assert (wrapped >= -180).all() and (wrapped < 180).all()
    np.testing.assert_allclose(wrapped, -180.0)


def test_wrap_degrees_preserves_scalar_and_dtype() -> None:
    """Scalar input gives a scalar, and float32 input stays float32."""
    assert np.ndim(wrap_degrees(190.0)) == 0
    assert wrap_degrees(np.array([190.0], dtype=np.float32)).dtype == np.float32
