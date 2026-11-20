"""scan_pose_metadata: metadata-only scan of unified and legacy pose files."""

import dataclasses
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from jabs.core.enums import PoseCapability
from jabs.core.utils import hash_file
from jabs.io.internal.pose_file import ComponentReader
from jabs.io.internal.pose_file import reader as pose_file_reader
from jabs.pose_estimation import PoseOpenError, open_pose_file
from jabs.pose_estimation.conversion_cache import ensure_converted
from jabs.pose_estimation.metadata import PoseMetadata, scan_pose_metadata

from .conftest import build_pose_file
from .contract import sample_path

_LEGACY_CORPUS = (
    Path(__file__).parents[2] / "packages" / "jabs-io" / "tests" / "data" / "legacy-pose"
)

# (source file, the legacy name it is given)
_SAMPLES = [
    (sample_path(2), "vid_pose_est_v2.h5"),
    (sample_path(3), "vid_pose_est_v3.h5"),
    (sample_path(4), "vid_pose_est_v4.h5"),
    (sample_path(5), "vid_pose_est_v5.h5"),
    (sample_path(6), "vid_pose_est_v6.h5"),
    (_LEGACY_CORPUS / "v3-tracklets.h5", "vid_pose_est_v3.h5"),
    (_LEGACY_CORPUS / "v7-dynamic-objects.h5", "vid_pose_est_v7.h5"),
    (_LEGACY_CORPUS / "v8-with-centers.h5", "vid_pose_est_v8.h5"),
]
_V3_SAMPLES = [s for s in _SAMPLES if s[1] == "vid_pose_est_v3.h5"]


def _legacy_and_converted(tmp_path: Path, source: Path, name: str) -> tuple[Path, Path]:
    """Copy a legacy sample under ``name`` and its conversion renamed ``vid_pose.h5``."""
    legacy_dir = tmp_path / "legacy"
    unified_dir = tmp_path / "unified"
    legacy_dir.mkdir()
    unified_dir.mkdir()
    legacy = legacy_dir / name
    shutil.copyfile(source, legacy)
    converted = ensure_converted(legacy, hash_file(legacy), tmp_path / "cache")
    unified = unified_dir / "vid_pose.h5"
    shutil.copyfile(converted, unified)
    return legacy, unified


# Keypoint datasets of a legacy and of a unified pose file.
_KEYPOINT_DATASETS = ("/poseest/points", "/poseest/confidence", "/jabs/pose/points")


def _refuse_keypoint_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any component read, or read of a keypoint dataset, raise."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the scan read component data")

    monkeypatch.setattr(pose_file_reader, "read_component", refuse)
    monkeypatch.setattr(ComponentReader, "read", refuse)

    real_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(self: h5py.Dataset, *args: object, **kwargs: object) -> object:
        if self.name in _KEYPOINT_DATASETS:
            raise AssertionError(f"the scan read {self.name}")
        return real_getitem(self, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)


@pytest.mark.parametrize(("source", "name"), _SAMPLES, ids=lambda v: getattr(v, "name", v))
def test_legacy_and_its_conversion_agree(tmp_path: Path, source: Path, name: str) -> None:
    """A legacy file and its conversion scan alike; only source_format follows the name."""
    legacy, unified = _legacy_and_converted(tmp_path, source, name)

    legacy_meta = scan_pose_metadata(legacy)
    unified_meta = scan_pose_metadata(unified)

    assert dataclasses.replace(legacy_meta, source_format="") == dataclasses.replace(
        unified_meta, source_format=""
    )
    assert legacy_meta.source_format == name.removeprefix("vid_").removesuffix(".h5")
    assert unified_meta.source_format == "jabs.pose-file"


@pytest.mark.parametrize(("source", "name"), _SAMPLES, ids=lambda v: getattr(v, "name", v))
def test_scan_matches_the_opened_pose(tmp_path: Path, source: Path, name: str) -> None:
    """Scanned metadata matches what Pose reports for the converted file."""
    _, unified = _legacy_and_converted(tmp_path, source, name)
    meta = scan_pose_metadata(unified)
    with open_pose_file(unified) as pose:
        assert meta.frames == pose.num_frames
        assert meta.num_identities == pose.num_identities
        assert meta.capabilities == pose.capabilities
        assert meta.cm_per_pixel == pose.cm_per_pixel
        assert meta.static_objects == tuple(sorted(pose.static_objects))
        assert meta.lixit_keypoints == pose.num_lixit_keypoints


@pytest.mark.parametrize(("source", "name"), _V3_SAMPLES, ids=lambda v: getattr(v, "name", v))
def test_v3_identities_are_its_track_slots(
    tmp_path: Path, source: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both v3 samples: the scan counts what open_pose_file counts, legacy and unified.

    ``open_pose_file`` opens the legacy file as ``Pose`` (``JABS_FORCE_UNIFIED``):
    the legacy v3 reader counts ``points.shape[1]``, which for v3-tracklets is
    fewer than the track slots the identities rule gives.
    """
    legacy, unified = _legacy_and_converted(tmp_path, source, name)
    monkeypatch.setenv("JABS_FORCE_UNIFIED", "1")
    with open_pose_file(legacy, tmp_path / "cache") as pose:
        expected = pose.num_identities
    assert expected > 0
    assert scan_pose_metadata(legacy).num_identities == expected
    assert scan_pose_metadata(unified).num_identities == expected


@pytest.mark.parametrize(("source", "name"), _SAMPLES, ids=lambda v: getattr(v, "name", v))
def test_scan_reads_no_keypoints(
    tmp_path: Path, source: Path, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither kind of scan reads component or keypoint data."""
    legacy, unified = _legacy_and_converted(tmp_path, source, name)
    _refuse_keypoint_reads(monkeypatch)
    # the guard does fire on the keypoint datasets of both kinds of file
    with h5py.File(legacy, "r") as h5, pytest.raises(AssertionError):
        h5["poseest/points"][0]
    with h5py.File(unified, "r") as h5, pytest.raises(AssertionError):
        h5["jabs/pose/points"][0]
    scan_pose_metadata(legacy)
    scan_pose_metadata(unified)


def test_unified_capabilities_from_the_manifest(tmp_path: Path) -> None:
    """Segmentation, boxes and social come from the component list and identities."""
    path = build_pose_file(
        tmp_path / "vid_pose.h5",
        slots=3,
        identity=3,
        segmentation=True,
        bbox=True,
        cm_per_pixel=0.5,
        static_objects={"lixit": np.zeros((1, 3, 2), dtype=np.float32)},
    )
    meta = scan_pose_metadata(path)
    assert meta == PoseMetadata(
        frames=6,
        num_identities=3,
        static_objects=("lixit",),
        lixit_keypoints=3,
        cm_per_pixel=0.5,
        capabilities=frozenset(
            {PoseCapability.SOCIAL, PoseCapability.SEGMENTATION, PoseCapability.BOUNDING_BOXES}
        ),
        source_format="jabs.pose-file",
    )


def test_single_identity_is_not_social(tmp_path: Path) -> None:
    """One identity gives no SOCIAL capability."""
    path = build_pose_file(tmp_path / "vid_pose.h5", slots=1, identity=1)
    assert scan_pose_metadata(path).capabilities == frozenset()


def test_unconvertible_legacy_file_raises_pose_open_error(tmp_path: Path) -> None:
    """A legacy file the converter refuses raises PoseOpenError naming the file."""
    path = tmp_path / "vid_pose_est_v8.h5"
    shutil.copyfile(_LEGACY_CORPUS / "v8-unknown-dataset.h5", path)
    with pytest.raises(PoseOpenError) as excinfo:
        scan_pose_metadata(path)
    assert excinfo.value.path == path
    assert excinfo.value.reason


def test_not_a_pose_file_raises_pose_open_error(tmp_path: Path) -> None:
    """A unified-named file that is not a pose file raises PoseOpenError."""
    path = tmp_path / "vid_pose.h5"
    with h5py.File(path, "w") as h5:
        h5["x"] = 1
    with pytest.raises(PoseOpenError):
        scan_pose_metadata(path)


def test_missing_file_raises_pose_open_error(tmp_path: Path) -> None:
    """An unreadable file raises PoseOpenError."""
    with pytest.raises(PoseOpenError):
        scan_pose_metadata(tmp_path / "vid_pose_est_v6.h5")


def test_unknown_name_raises_pose_open_error(tmp_path: Path) -> None:
    """A name following neither convention raises PoseOpenError."""
    path = tmp_path / "vid.h5"
    shutil.copyfile(sample_path(4), path)
    with pytest.raises(PoseOpenError):
        scan_pose_metadata(path)


def test_malformed_manifest_raises_pose_open_error(tmp_path: Path) -> None:
    """A manifest that passes the schema but breaks the scan (a 1-D lixit) is a PoseOpenError."""
    path = build_pose_file(
        tmp_path / "vid_pose.h5", static_objects={"lixit": np.zeros((1, 3, 2), dtype=np.float32)}
    )
    with h5py.File(path, "a") as h5:
        manifest = json.loads(h5["manifest"][()])
        for spec in manifest["components"]:
            if spec["id"].endswith("lixit"):
                spec["shape"] = [1]
        del h5["manifest"]
        h5["manifest"] = json.dumps(manifest)
    with pytest.raises(PoseOpenError) as excinfo:
        scan_pose_metadata(path)
    assert excinfo.value.path == path
    assert excinfo.value.reason
