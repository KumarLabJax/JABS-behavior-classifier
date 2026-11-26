"""Shared paths and helpers for the legacy converter tests.

The synthetic corpus lives in ``tests/data/legacy-pose`` and is regenerated
with ``uv run python packages/jabs-io/tests/data/legacy-pose/generate.py``.
The real public samples live in the repository root's ``tests/data``, outside
this package, so tests that use them skip when they are absent.
"""

import importlib.util
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np
import pytest

LEGACY = Path(__file__).parents[3] / "data" / "legacy-pose"

# packages/jabs-io/tests/internal/pose_file/convert -> the repository root.
SAMPLES = Path(__file__).parents[6] / "tests" / "data"

requires_samples = pytest.mark.skipif(
    not (SAMPLES / "sample_pose_est_v4.h5").exists(),
    reason="the public legacy samples are not in this checkout",
)


def legacy(name: str) -> Path:
    """Return the path of one synthetic legacy corpus file.

    Args:
        name: The file's name, such as ``"v8-edge-cases.h5"``.

    Returns:
        Its path in the committed corpus.
    """
    return LEGACY / name


@cache
def generator() -> ModuleType:
    """Load the corpus generator, so tests can call ``write_legacy`` themselves.

    It is a script beside the data, not a package module, so it is loaded by
    path. The distinct module name keeps it apart from the pose-format
    generator in ``sys.modules``.

    Returns:
        The ``generate.py`` module.
    """
    spec = importlib.util.spec_from_file_location("legacy_pose_generate", LEGACY / "generate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def widen_segmentation(
    extra: int, contour_frame: int | None = None
) -> Callable[[h5py.File], None]:
    """Return an edit giving the segmentation datasets ``extra`` more rows than ``points``.

    Real v6 files have this layout: the segmentation model writes its own
    detection axis, one row wider than the pose's, and the spare rows hold no
    contours. ``instance_seg_id`` is ``index + 1`` on every row, as those files
    have it; the spare rows of the other datasets are empty.

    Args:
        extra: Rows to add past ``points.shape[1]``.
        contour_frame: When given, the first spare row holds a contour in this
            frame, which no pose instance can carry.

    Returns:
        An edit for ``write_legacy``.
    """

    def edit(h5: h5py.File) -> None:
        """Replace the four segmentation datasets with wider ones."""
        pose = h5["poseest"]
        seg = pose["seg_data"][...]
        frames, instances = seg.shape[:2]
        wide = np.full((frames, instances + extra, *seg.shape[2:]), -1, dtype=seg.dtype)
        wide[:, :instances] = seg
        if contour_frame is not None:
            wide[contour_frame, instances, 0, :3] = [[500, 200], [510, 200], [505, 210]]
        flag = np.zeros(wide.shape[:3], dtype=bool)
        flag[:, :instances] = pose["seg_external_flag"][...]
        longterm = np.zeros((frames, instances + extra), dtype=np.uint32)
        longterm[:, :instances] = pose["longterm_seg_id"][...]
        seg_id = np.tile(np.arange(1, instances + extra + 1, dtype=np.uint32), (frames, 1))
        for name, data in (
            ("seg_data", wide),
            ("seg_external_flag", flag),
            ("longterm_seg_id", longterm),
            ("instance_seg_id", seg_id),
        ):
            del pose[name]
            pose.create_dataset(name, data=data)

    return edit
