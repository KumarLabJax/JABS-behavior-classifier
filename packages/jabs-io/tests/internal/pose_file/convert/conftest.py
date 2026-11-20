"""Shared paths and helpers for the legacy converter tests.

The synthetic corpus lives in ``tests/data/legacy-pose`` and is regenerated
with ``uv run python packages/jabs-io/tests/data/legacy-pose/generate.py``.
The real public samples live in the repository root's ``tests/data``, outside
this package, so tests that use them skip when they are absent.
"""

import importlib.util
from functools import cache
from pathlib import Path
from types import ModuleType

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
