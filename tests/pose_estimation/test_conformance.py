"""Every legacy file JABS reads opens through the conversion cache as a conforming file."""

import importlib.util
import shutil
from functools import cache
from pathlib import Path
from types import ModuleType

import h5py
import pytest

from jabs.core.utils import hash_file
from jabs.io.internal.pose_file import validate
from jabs.pose_estimation import open_pose_file
from jabs.pose_estimation.conversion_cache import converted_path, open_unified

from .contract import sample_path

_LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)

# the repository's public samples, and the converter's synthetic corpus files it
# converts (the refusal files are covered by test_conversion_cache)
_SAMPLES = [(sample_path(v), v) for v in (2, 3, 4, 5, 6)]
_CORPUS = [
    (_LEGACY_CORPUS / name, version)
    for name, version in (
        ("v7-dynamic-objects.h5", 7),
        ("v8-with-centers.h5", 8),
        ("v8-without-centers.h5", 8),
        ("v8-edge-cases.h5", 8),
        ("v8-no-contours.h5", 8),
    )
]


@cache
def _generator() -> ModuleType:
    """Load the jabs-io legacy corpus generator script by path."""
    spec = importlib.util.spec_from_file_location(
        "jabs_conformance_legacy_generate", _LEGACY_CORPUS / "generate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("source", "version"), _SAMPLES + _CORPUS, ids=lambda v: v.stem if isinstance(v, Path) else ""
)
def test_legacy_file_opens_and_conversion_validates(
    tmp_path: Path, source: Path, version: int
) -> None:
    """A legacy file opens with open_unified; its conversion passes the deep check."""
    path = tmp_path / f"{source.stem.replace('-', '_')}_pose_est_v{version}.h5"
    shutil.copyfile(source, path)
    cache_dir = tmp_path / "cache"
    with open_unified(path, cache_dir) as pose:
        assert pose.num_frames > 0
        assert pose.source_format == f"pose_est_v{version}"
        assert pose.hash == hash_file(path)
    converted = converted_path(path, hash_file(path), cache_dir)
    findings = validate(converted, deep=True)
    assert [f for f in findings if f.severity == "error"] == []


def _write_all_masked_v4(path: Path) -> Path:
    """Write a v4 file whose instances are all masked, keeping their confidences."""
    gen = _generator()

    def mask_everything(h5: h5py.File) -> None:
        """Mask every instance, keeping its confidences."""
        del h5["poseest/instance_id_center"]
        h5["poseest/id_mask"][...] = True

    gen.write_legacy(path, **gen.SMALL, version=(4, 0), seg=False, edits=mask_everything)
    with h5py.File(path, "r") as h5:
        assert h5["poseest/confidence"][()].max() > 0
    return path


def test_all_masked_v4_has_no_identities(tmp_path: Path) -> None:
    """A v4 file whose instances are all masked, though confident, has no identities.

    Without ``instance_id_center`` the identity count comes from valid embed ids,
    and there are none, so the conversion has identity = 0. It keeps the v4
    tracklets, which must not turn its slots into identities (only a v3
    conversion's do).
    """
    path = _write_all_masked_v4(tmp_path / "vid_pose_est_v4.h5")
    with open_unified(path, tmp_path / "cache") as pose:
        assert pose.identities == []
        assert pose.num_identities == 0


@pytest.mark.legacy_reader
def test_all_masked_v4_legacy_reader_agrees(tmp_path: Path) -> None:
    """The legacy v4 reader also finds no identities in an all-masked v4 file."""
    path = _write_all_masked_v4(tmp_path / "vid_pose_est_v4.h5")
    with open_pose_file(path, tmp_path) as legacy:
        assert legacy.identities == []
