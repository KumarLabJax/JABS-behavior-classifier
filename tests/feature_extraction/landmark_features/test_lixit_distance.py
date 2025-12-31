"""Tests for lixit distance feature extraction."""

import numpy as np
import pytest


def test_dimensions(pose_est_v5_with_static_objects, lixit_distance_features):
    """Test dimensions of per frame and window feature values.

    Verifies that lixit distance features have the correct shape.
    """
    _, lixit_distance = lixit_distance_features

    # check dimensions of per frame feature values
    for i in range(pose_est_v5_with_static_objects.num_identities):
        distances = lixit_distance.per_frame(i)

        # Check that we got a dictionary back
        assert isinstance(distances, dict)

        # Check dimensions for NOSE keypoint (could check all keypoints)
        assert "distance to lixit NOSE" in distances
        assert distances["distance to lixit NOSE"].shape == (
            pose_est_v5_with_static_objects.num_frames,
        )

        # check dimensions of window feature values
        dist_window_values = lixit_distance.window(i, 5, distances)
        for op in dist_window_values:
            assert dist_window_values[op]["distance to lixit NOSE"].shape == (
                pose_est_v5_with_static_objects.num_frames,
            )


def test_distances_greater_equal_zero(pose_est_v5_with_static_objects, lixit_distance_features):
    """Test that computed distances are non-negative."""
    _, lixit_distance = lixit_distance_features

    for i in range(pose_est_v5_with_static_objects.num_identities):
        distances = lixit_distance.per_frame(i)

        # Check all distance features are >= 0 (ignoring NaN values)
        for feature_name, feature_values in distances.items():
            valid_distances = feature_values[~np.isnan(feature_values)]
            assert (valid_distances >= 0).all(), f"{feature_name} has negative distances"


def test_computation(lixit_distance_features):
    """Test lixit distance computation against expected values.

    Spot checks distance values for identity 0 against known results.
    These values have been verified to match manual Euclidean distance
    calculations between nose keypoint and lixit position [62, 166]
    with pixel scaling applied.
    """
    _, lixit_distance = lixit_distance_features

    # spot check some distance values for identity 0
    # Note: These expected values are for the NOSE keypoint distance to lixit
    # with lixit at position [62, 166] and pixel scale applied
    expected = np.asarray(
        [
            18.4560733207,
            18.5266151431,
            18.5773552151,
            18.6480151524,
            18.7383696105,
            18.7583096218,
            18.7583096218,
            18.8288618427,
            18.8288618427,
            18.9900097824,
        ],
        dtype=np.float32,
    )

    distances = lixit_distance.per_frame(0)
    actual = distances["distance to lixit NOSE"]

    # Compare first 10 values
    for i in range(expected.shape[0]):
        assert expected[i] == pytest.approx(actual[i], abs=1e-5)
