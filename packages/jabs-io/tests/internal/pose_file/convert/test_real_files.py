"""Opt-in: every real legacy pose file in ``JABS_LEGACY_POSE_DIR`` converts cleanly.

Real files are large and not in the repository, so this runs only when the
variable names a directory of ``*_pose_est_v*.h5`` files; a directory with
none fails rather than passing on nothing. The converter covers ``v2``..``v8``,
so any refusal or an error from the deep check fails.
"""

import os
from pathlib import Path

import pytest

from jabs.io.internal.pose_file.convert import convert_legacy
from jabs.io.internal.pose_file.validate import validate

_DIRECTORY = os.environ.get("JABS_LEGACY_POSE_DIR")

_requires_directory = pytest.mark.skipif(not _DIRECTORY, reason="JABS_LEGACY_POSE_DIR is not set")


def _real_files() -> list[Path]:
    """The legacy files in ``JABS_LEGACY_POSE_DIR``, in a stable order."""
    if not _DIRECTORY:
        return []
    return sorted(Path(_DIRECTORY).glob("*_pose_est_v*.h5"))


@_requires_directory
def test_real_files_are_found() -> None:
    """The directory holds legacy files, so the parametrized test is not silently empty."""
    assert _real_files(), f"JABS_LEGACY_POSE_DIR={_DIRECTORY} holds no *_pose_est_v*.h5 files"


@_requires_directory
@pytest.mark.parametrize("source", _real_files(), ids=lambda path: path.name)
def test_real_files(tmp_path: Path, source: Path) -> None:
    """The file converts and the deep check finds no errors."""
    out = tmp_path / "out.h5"
    convert_legacy(source, out)

    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []
