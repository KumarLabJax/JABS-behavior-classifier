"""The deprecated ``jabs.core.abstract.PoseEstimation`` name.

The legacy pose reader ABC is gone. Its name stays for one release as a class
that only carries the skeleton constants, and every access to it warns.
"""

import importlib
import json
import warnings

import numpy as np
import pytest

from jabs.core import pose_skeleton
from jabs.core.pose_protocol import parse_model_metadata


def test_package_name_warns() -> None:
    """``from jabs.core.abstract import PoseEstimation`` warns."""
    with pytest.warns(DeprecationWarning, match="PoseEstimation"):
        from jabs.core.abstract import PoseEstimation  # noqa: F401


def test_module_name_warns() -> None:
    """``from jabs.core.abstract.pose_est import PoseEstimation`` warns."""
    with pytest.warns(DeprecationWarning, match="PoseEstimation"):
        from jabs.core.abstract.pose_est import PoseEstimation  # noqa: F401


def test_every_access_warns() -> None:
    """The name is never cached in the module, so a second access warns again."""
    pose_est = importlib.import_module("jabs.core.abstract.pose_est")
    with pytest.warns(DeprecationWarning):
        pose_est.PoseEstimation  # noqa: B018
    with pytest.warns(DeprecationWarning):
        pose_est.PoseEstimation  # noqa: B018


def test_package_access_warns_once() -> None:
    """Going through the package emits one warning, not one per module."""
    abstract = importlib.import_module("jabs.core.abstract")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        abstract.PoseEstimation  # noqa: B018
    assert [w.category for w in caught] == [DeprecationWarning]


@pytest.mark.parametrize("module", ["jabs.core.abstract", "jabs.core.abstract.pose_est"])
def test_from_import_warns_once_at_the_import(module: str) -> None:
    """A ``from ... import PoseEstimation`` statement warns once, attributed to itself."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(f"from {module} import PoseEstimation", {})
    assert [w.category for w in caught] == [DeprecationWarning]
    assert caught[0].filename == "<string>"


def test_importing_the_package_does_not_warn() -> None:
    """Importing ``jabs.core.abstract`` or its module, or MINIMUM_CONFIDENCE, is silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.reload(importlib.import_module("jabs.core.abstract"))
        pose_est = importlib.reload(importlib.import_module("jabs.core.abstract.pose_est"))
        assert pose_est.MINIMUM_CONFIDENCE == pose_skeleton.MINIMUM_CONFIDENCE


def test_deprecated_class_carries_only_the_skeleton_constants() -> None:
    """The class attributes are the same objects as ``jabs.core.pose_skeleton``'s."""
    with pytest.warns(DeprecationWarning):
        from jabs.core.abstract import PoseEstimation
    assert PoseEstimation.KeypointIndex is pose_skeleton.KeypointIndex
    assert PoseEstimation.FULL_CONNECTED_SEGMENTS is pose_skeleton.FULL_CONNECTED_SEGMENTS
    assert PoseEstimation.NVSN_CONNECTED_SEGMENTS is pose_skeleton.NVSN_CONNECTED_SEGMENTS
    public = {name for name in vars(PoseEstimation) if not name.startswith("_")}
    assert public == {"KeypointIndex", "FULL_CONNECTED_SEGMENTS", "NVSN_CONNECTED_SEGMENTS"}


def test_unknown_names_still_raise() -> None:
    """Module ``__getattr__`` raises AttributeError for any other name."""
    abstract = importlib.import_module("jabs.core.abstract")
    pose_est = importlib.import_module("jabs.core.abstract.pose_est")
    with pytest.raises(AttributeError):
        abstract.NoSuchName  # noqa: B018
    with pytest.raises(AttributeError):
        pose_est.NoSuchName  # noqa: B018


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        (b'{"a": 1}', {"a": 1}),
        (np.array(json.dumps({"a": [1, 2]})), {"a": [1, 2]}),
        ("not json", None),
        ("[1, 2]", None),
    ],
    ids=["absent", "empty", "blank", "bytes", "0-d array", "invalid", "not an object"],
)
def test_parse_model_metadata(raw: object, expected: object) -> None:
    """parse_model_metadata lives in jabs.core.pose_protocol now."""
    assert parse_model_metadata(raw) == expected
