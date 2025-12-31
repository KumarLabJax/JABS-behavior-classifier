"""Shared fixtures for feature module tests."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jabs.pose_estimation
from jabs.feature_extraction.landmark_features.food_hopper import FoodHopper


@pytest.fixture(scope="module")
def pose_est_v5_with_static_objects():
    """Fixture to create a pose estimation v5 object with static objects.

    Creates a temporary directory with a copy of the pose file and adds
    lixit and food hopper static objects to it.

    Yields:
        PoseEstimationV5: Pose estimation object with static objects added.
    """
    test_file = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v5.h5"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        pose_path = tmpdir_path / "sample_pose_est_v5.h5"

        # Copy pose file to tempdir
        shutil.copy(test_file, pose_path)

        pose_est_v5 = jabs.pose_estimation.open_pose_file(pose_path)

        # V5 pose file in the data directory does not currently have "lixit"
        # as one of its static objects, so we'll manually add it
        pose_est_v5.static_objects["lixit"] = np.asarray([[62, 166]], dtype=np.uint16)

        # V5 pose file also doesn't have the food hopper static object
        pose_est_v5.static_objects["food_hopper"] = np.asarray(
            [[7, 291], [7, 528], [44, 296], [44, 518]]
        )

        yield pose_est_v5


@pytest.fixture(scope="module")
def food_hopper_feature(pose_est_v5_with_static_objects):
    """Fixture to create a FoodHopper feature instance.

    Args:
        pose_est_v5_with_static_objects: Pose estimation fixture.

    Returns:
        FoodHopper: Food hopper feature instance.
    """
    pixel_scale = pose_est_v5_with_static_objects.cm_per_pixel
    return FoodHopper(pose_est_v5_with_static_objects, pixel_scale)


@pytest.fixture(scope="module")
def lixit_distance_features(pose_est_v5_with_static_objects):
    """Fixture to create lixit distance feature instances.

    Args:
        pose_est_v5_with_static_objects: Pose estimation fixture.

    Returns:
        tuple: (LixitDistanceInfo, DistanceToLixit) instances.
    """
    from jabs.feature_extraction.landmark_features import lixit

    pixel_scale = pose_est_v5_with_static_objects.cm_per_pixel
    distance_info = lixit.LixitDistanceInfo(pose_est_v5_with_static_objects, pixel_scale)
    lixit_distance = lixit.DistanceToLixit(
        pose_est_v5_with_static_objects, pixel_scale, distance_info
    )
    return distance_info, lixit_distance
