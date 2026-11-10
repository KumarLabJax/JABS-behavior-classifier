"""Tests for jabs.core.utils.static_objects."""

import numpy as np
import pytest

from jabs.core.utils.static_objects import lixit_keypoint_count


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((4, 3, 2), 3),
        ((1, 3, 2), 3),
        ((4, 2), 1),
        ((1, 2), 1),
        ((0, 3, 2), 0),
        ((0, 2), 0),
        ((), 0),
    ],
    ids=[
        "three-keypoint",
        "three-keypoint-single",
        "one-keypoint",
        "one-keypoint-single",
        "empty-three-keypoint",
        "empty-one-keypoint",
        "scalar",
    ],
)
def test_lixit_keypoint_count(shape: tuple[int, ...], expected: int) -> None:
    """The keypoint count comes from the dataset's dimensionality, and 0 means empty."""
    assert lixit_keypoint_count(shape) == expected


def test_lixit_keypoint_count_accepts_an_array_shape() -> None:
    """A numpy array's ``.shape`` works as well as an h5py dataset's."""
    assert lixit_keypoint_count(np.zeros((2, 3, 2)).shape) == 3
