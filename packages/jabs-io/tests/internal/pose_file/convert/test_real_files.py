"""Opt-in: every real legacy pose file in ``JABS_LEGACY_POSE_DIR`` converts cleanly.

Real files are large and not in the repository, so this runs only when the
variable names a directory of ``*_pose_est_v*.h5`` files; a directory with
none fails rather than passing on nothing. Real directories mix versions, and
the converter covers ``v4``..``v8`` by decision, so a file lacking **both**
``poseest/id_mask`` and ``poseest/instance_embed_id`` predates v4 and is
skipped. A damaged v4+ file missing only one of them fails, as does any other
refusal or an error from the deep check.
"""

import os
from pathlib import Path

import h5py
import pytest

from jabs.io.internal.pose_file.convert import convert_legacy
from jabs.io.internal.pose_file.validate import validate

_DIRECTORY = os.environ.get("JABS_LEGACY_POSE_DIR")

# The datasets v4 introduced; a file with neither is older.
_V4_DATASETS = ("id_mask", "instance_embed_id")

_requires_directory = pytest.mark.skipif(not _DIRECTORY, reason="JABS_LEGACY_POSE_DIR is not set")


def _real_files() -> list[Path]:
    """The legacy files in ``JABS_LEGACY_POSE_DIR``, in a stable order."""
    if not _DIRECTORY:
        return []
    return sorted(Path(_DIRECTORY).glob("*_pose_est_v*.h5"))


def _predates_v4(path: Path) -> bool:
    """Return whether a legacy file has none of the datasets v4 introduced."""
    with h5py.File(path, "r") as h5:
        # Without poseest it is no legacy pose file at all, which must fail.
        pose = h5.get("poseest")
        return pose is not None and not any(name in pose for name in _V4_DATASETS)


@_requires_directory
def test_real_files_are_found() -> None:
    """The directory holds legacy files, so the parametrized test is not silently empty."""
    assert _real_files(), f"JABS_LEGACY_POSE_DIR={_DIRECTORY} holds no *_pose_est_v*.h5 files"


@_requires_directory
@pytest.mark.parametrize("source", _real_files(), ids=lambda path: path.name)
def test_real_files(tmp_path: Path, source: Path) -> None:
    """The file converts and the deep check finds no errors, unless it predates v4."""
    if _predates_v4(source):
        pytest.skip(f"{source.name} has neither poseest/id_mask nor poseest/instance_embed_id")

    out = tmp_path / "out.h5"
    convert_legacy(source, out)

    assert [f for f in validate(out, deep=True) if f.severity == "error"] == []
