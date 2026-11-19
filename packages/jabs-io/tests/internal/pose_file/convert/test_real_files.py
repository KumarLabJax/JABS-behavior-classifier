"""Opt-in: every real legacy pose file in ``JABS_LEGACY_POSE_DIR`` converts cleanly.

Real files are large and not in the repository, so this runs only when the
variable names a directory of ``*_pose_est_v*.h5`` files. Real directories mix
versions, and the converter covers ``v4``..``v8`` by decision, so a file that
``inspect_legacy`` refuses as older than ``v4`` is skipped. Any other refusal,
or an error from the deep check, fails.
"""

import os
from pathlib import Path

import pytest

from jabs.io.internal.pose_file.convert import (
    LegacyConversionError,
    convert_legacy,
    inspect_legacy,
)
from jabs.io.internal.pose_file.validate import validate

_DIRECTORY = os.environ.get("JABS_LEGACY_POSE_DIR")

# The refusal inspect_legacy makes for a file without the v4 identity datasets.
_BELOW_V4 = "not a pose_est_v4 or later file"


def _real_files() -> list[Path]:
    """The legacy files in ``JABS_LEGACY_POSE_DIR``, in a stable order."""
    if not _DIRECTORY:
        return []
    return sorted(Path(_DIRECTORY).glob("*_pose_est_v*.h5"))


@pytest.mark.skipif(not _DIRECTORY, reason="JABS_LEGACY_POSE_DIR is not set")
@pytest.mark.parametrize("source", _real_files(), ids=lambda path: path.name)
def test_real_files(tmp_path: Path, source: Path) -> None:
    """The file converts and the deep check finds no errors, unless it predates v4."""
    try:
        inspect_legacy(source)
    except LegacyConversionError as error:
        if _BELOW_V4 not in str(error):
            raise
        pytest.skip(str(error))

    out = tmp_path / "out.h5"
    convert_legacy(source, out)

    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []
