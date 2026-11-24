"""PoseLike contract tests for ``Pose`` over converted copies of the legacy samples.

The goldens were recorded from the legacy readers (``test_contract_legacy.py``).
``Pose`` must match them except where spec §2 (docs/superpowers/specs/
2026-11-20-jabs-adopts-pose-file-design.md) changes the behavior. Each allowed
difference is applied to a copy of the golden by a named function below; nothing
else may differ.

Allowed differences:

* v3 bearings are NaN on frames where BASE_NECK or BASE_TAIL is invalid; the
  legacy v3 reader computed them from the stored (0, 0). Spec §2 row "Point
  validity": invalid points are NaN.
* v3 identity masks may differ through the >= 3 valid body keypoints rule (spec
  §2 row "Identity present"; legacy v3 needed 1). On the samples there is no such
  difference, so no adjustment is made and the masks are compared exactly.
"""

import shutil
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from jabs.core.pose_protocol import PoseLike
from jabs.core.pose_skeleton import KeypointIndex
from jabs.pose_estimation.conversion_cache import open_unified
from jabs.pose_estimation.pose import Pose

from .contract import (
    SAMPLE_VERSIONS,
    check_pose_contract,
    golden_path,
    load_golden,
    sample_path,
)

_BEARING_KEYPOINTS = (KeypointIndex.BASE_NECK.value, KeypointIndex.BASE_TAIL.value)


def _v3_bearings_nan_where_invalid(
    expected: dict[str, npt.NDArray[Any]],
) -> tuple[dict[str, npt.NDArray[Any]], int]:
    """Spec §2 "Point validity": NaN bearings where BASE_NECK or BASE_TAIL is invalid.

    Returns:
        The adjusted golden copy and how many (identity, frame) bearings changed.
    """
    adjusted = dict(expected)
    changed = 0
    for key in [k for k in expected if k.endswith("/bearings")]:
        prefix = key.removesuffix("/bearings")
        point_mask = expected[f"{prefix}/point_mask"]
        invalid = ~point_mask[:, _BEARING_KEYPOINTS].all(axis=1)
        bearings = expected[key].copy()
        changed += int(np.count_nonzero(invalid & ~np.isnan(bearings)))
        bearings[invalid] = np.nan
        adjusted[key] = bearings
    return adjusted, changed


@pytest.fixture(scope="module", params=SAMPLE_VERSIONS, ids=lambda v: f"v{v}")
def unified_sample(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[int, Pose]]:
    """Open a copy of each sample legacy file through open_unified, once per module."""
    version = request.param
    root = tmp_path_factory.mktemp(f"unified_v{version}")
    source = root / sample_path(version).name
    shutil.copyfile(sample_path(version), source)
    with open_unified(source, root / "cache") as pose:
        yield version, pose


def test_pose_satisfies_protocol(unified_sample: tuple[int, Pose]) -> None:
    """Pose is structurally a PoseLike."""
    _, pose = unified_sample
    assert isinstance(pose, PoseLike)


def test_unified_matches_golden(unified_sample: tuple[int, Pose]) -> None:
    """Pose over a converted sample matches the legacy goldens, with the listed exceptions."""
    version, pose = unified_sample
    expected = load_golden(golden_path(version))
    if version == 3:
        expected, changed = _v3_bearings_nan_where_invalid(expected)
        # the allowance must describe a real difference, not pass vacuously
        assert changed > 0
    check_pose_contract(pose, expected)
