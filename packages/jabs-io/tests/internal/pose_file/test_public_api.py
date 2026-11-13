"""The module surface, and proof the legacy adapter is unaffected."""

import numpy as np
import pytest

from jabs.core.enums import JabsPoseVersion
from jabs.core.types.pose import PoseData
from jabs.io.internal.pose import hdf5 as pose_hdf5
from jabs.io.internal.pose_file import (
    Component,
    Finding,
    NotAPoseFileError,
    PoseFile,
    Provenance,
    ProvenanceRecord,
    Skeleton,
    VideoInfo,
    read_component,
    read_manifest,
    read_pose_file,
    validate,
    write_pose_file,
)


def _pose_data() -> PoseData:
    frames = 3
    return PoseData(
        points=np.zeros((1, frames, 12, 2), dtype=np.float64),
        point_mask=np.ones((1, frames, 12), dtype=bool),
        identity_mask=np.ones((1, frames), dtype=bool),
        body_parts=[f"p{i}" for i in range(12)],
        edges=[(0, 1)],
        fps=30,
        # Legacy v2 requires confidence, and recovers point validity from it.
        confidence=np.ones((1, frames, 12), dtype=np.float32),
    )


def test_public_names_are_importable():
    """The package exposes one coherent surface."""
    for callable_name in (
        write_pose_file,
        read_pose_file,
        read_component,
        read_manifest,
        validate,
    ):
        assert callable(callable_name)
    for type_name in (
        PoseFile,
        Component,
        Skeleton,
        VideoInfo,
        Provenance,
        ProvenanceRecord,
        Finding,
        NotAPoseFileError,
    ):
        assert type_name is not None


def test_round_trip_through_the_public_surface(tmp_path, sample_pose_file):
    """Write then read using only the public names."""
    path = tmp_path / "surface_pose.h5"
    write_pose_file(sample_pose_file, path)
    assert [f for f in validate(path) if f.severity == "error"] == []
    back = read_pose_file(path)
    np.testing.assert_array_equal(
        back.component("jabs.pose.points").data,
        sample_pose_file.component("jabs.pose.points").data,
    )


def test_legacy_v2_write_still_works(tmp_path):
    """The legacy path is untouched by this work."""
    out = tmp_path / "legacy_pose_est_v2.h5"
    pose_hdf5.PoseHDF5Adapter().write(_pose_data(), out, legacy=JabsPoseVersion.V2)
    assert out.exists()


def test_new_format_path_is_declared_but_not_yet_mapped(tmp_path):
    """legacy=None selects the new format, which still needs the mapping."""
    with pytest.raises(NotImplementedError, match="PoseFile"):
        pose_hdf5.PoseHDF5Adapter().write(_pose_data(), tmp_path / "new_pose.h5", legacy=None)


def test_unsupported_legacy_version_still_rejected(tmp_path):
    """A legacy version nobody implemented is a ValueError, not a silent write."""
    with pytest.raises(ValueError, match="Unsupported legacy pose version"):
        pose_hdf5.PoseHDF5Adapter().write(
            _pose_data(), tmp_path / "v3_pose.h5", legacy=JabsPoseVersion.V3
        )
