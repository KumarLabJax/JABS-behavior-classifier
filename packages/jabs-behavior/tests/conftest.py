"""Pytest configuration for jabs-behavior tests."""

import numpy as np
import pytest


@pytest.fixture
def simple_behavior_vector():
    """Fixture providing a simple behavior vector for testing."""
    return np.array([0, 0, 1, 1, 1, 0, 0, 1, 0])


@pytest.fixture
def complex_behavior_vector():
    """Fixture providing a complex behavior vector for testing."""
    return np.array(
        [
            0,
            0,
            0,  # NOT_BEHAVIOR
            1,
            1,  # BEHAVIOR
            0,  # NOT_BEHAVIOR
            1,
            1,
            1,  # BEHAVIOR
            0,
            0,  # NOT_BEHAVIOR
            1,  # BEHAVIOR
            0,
            0,
            0,
            0,  # NOT_BEHAVIOR
        ]
    )


@pytest.fixture
def vector_with_none():
    """Fixture providing a behavior vector with NONE labels."""
    return np.array([-1, -1, 0, 0, 1, 1, -1, 0, 0])


@pytest.fixture
def rle_data():
    """Fixture providing valid RLE data."""
    return {
        "starts": np.array([0, 3, 6, 9]),
        "durations": np.array([3, 3, 3, 3]),
        "states": np.array([0, 1, 0, 1]),
    }
