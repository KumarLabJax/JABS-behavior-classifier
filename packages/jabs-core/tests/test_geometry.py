"""Unit tests for jabs.core.utils.geometry module."""

import numpy as np
import pytest

from jabs.core.utils import signed_angle_degrees


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
