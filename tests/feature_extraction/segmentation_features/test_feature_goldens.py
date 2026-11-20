"""The segmentation-dependent features on the sample v6 file match their goldens."""

import numpy as np
import pytest
from feature_goldens import compute_features, load_golden


@pytest.fixture(scope="module")
def computed(seg_data: dict) -> dict[str, np.ndarray]:
    """Features computed on the sample v6 file, keyed like the goldens."""
    return compute_features(seg_data["pose_est_v6"])


def test_feature_keys_match_golden(computed: dict[str, np.ndarray]) -> None:
    """Every recorded feature is computed, and nothing else."""
    assert sorted(computed) == sorted(load_golden())


def test_features_match_golden(computed: dict[str, np.ndarray]) -> None:
    """Each feature equals its recorded values, NaN where they were NaN."""
    golden = load_golden()
    mismatched = [
        key
        for key, expected in golden.items()
        if not np.allclose(computed[key], expected, rtol=1e-5, atol=1e-6, equal_nan=True)
    ]
    assert not mismatched, mismatched
