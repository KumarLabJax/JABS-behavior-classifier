"""Tests for linking pose NWB files to their source video."""

import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest

# The NWB helpers need the optional `nwb` extra; an unguarded import would fail
# collection and abort every other test in this package, not just these.
pytest.importorskip("pynwb")
pytest.importorskip("ndx_pose")
pytest.importorskip("ndx_jabs")
pytest.importorskip("ndx_multisubjects")

from pynwb import NWBHDF5IO, NWBFile

from jabs.core.abstract.pose_est import PoseEstimation
from jabs.core.types import PoseData
from jabs.io.internal.pose import (
    PoseNWBAdapter,
    link_external_video,
    nwb_video,
    read_video_link_info,
    relative_video_path,
)

NUM_FRAMES = 12
FPS = 30


@pytest.fixture
def pose_nwb(tmp_path: Path) -> Path:
    """Write a one-identity pose NWB file named ``clip.nwb`` in a ``nwb`` directory."""
    body_parts = [kpt.name for kpt in PoseEstimation.KeypointIndex]
    data = PoseData(
        points=np.zeros((1, NUM_FRAMES, len(body_parts), 2)),
        point_mask=np.ones((1, NUM_FRAMES, len(body_parts)), dtype=bool),
        identity_mask=np.ones((1, NUM_FRAMES), dtype=bool),
        body_parts=body_parts,
        edges=[],
        fps=FPS,
        external_ids=["mouse"],
    )
    out_dir = tmp_path / "nwb"
    out_dir.mkdir()
    PoseNWBAdapter().write(data, out_dir / "clip.nwb")
    # the per-identity writer names the file clip_mouse.nwb
    path = out_dir / "clip.nwb"
    (out_dir / "clip_mouse.nwb").rename(path)
    return path


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """Create a stand-in video file in a ``videos`` directory next to the NWB directory."""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    path = video_dir / "clip.mp4"
    path.write_bytes(b"not a real video")
    return path


def _datasets(path: Path) -> dict[str, np.ndarray]:
    """Return every dataset of an HDF5 file keyed by its path."""
    found: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda name, obj: found.__setitem__(name, obj[()])
            if isinstance(obj, h5py.Dataset)
            else None
        )
    return found


@pytest.mark.parametrize(
    ("nwb_file", "video_file", "expected"),
    [
        ("/d/clip.nwb", "/d/clip.mp4", "clip.mp4"),
        ("/d/nwb/clip.nwb", "/d/videos/clip.mp4", "../videos/clip.mp4"),
        ("/d/clip.nwb", "/d/videos/clip.mp4", "videos/clip.mp4"),
        ("/d/a/b/clip.nwb", "/d/clip.mp4", "../../clip.mp4"),
    ],
    ids=["same-dir", "sibling-dir", "subdir", "parent-dir"],
)
def test_relative_video_path(nwb_file: str, video_file: str, expected: str) -> None:
    """The stored path is relative to the directory of the NWB file."""
    assert relative_video_path(Path(nwb_file), Path(video_file)) == expected


def test_read_video_link_info_unlinked(pose_nwb: Path) -> None:
    """A freshly converted file reports its pose timing and no linked video."""
    info = read_video_link_info(pose_nwb)

    assert info.num_frames == NUM_FRAMES
    assert info.rate == pytest.approx(FPS)
    assert info.external_files == []


def test_read_video_link_info_rejects_file_without_pose(tmp_path: Path) -> None:
    """A file that holds no JABS pose data is rejected with a KeyError."""
    path = tmp_path / "plain.nwb"
    nwbfile = NWBFile(
        session_description="no pose here",
        identifier="plain",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)

    with pytest.raises(KeyError):
        read_video_link_info(path)


def test_link_external_video_adds_image_series(pose_nwb: Path, video: Path) -> None:
    """The ImageSeries references the video externally and shares the pose timeline."""
    stored = link_external_video(
        pose_nwb, video, num_frames=NUM_FRAMES, rate=FPS, dimension=(640, 480)
    )

    assert stored == "../videos/clip.mp4"
    assert read_video_link_info(pose_nwb).external_files == ["../videos/clip.mp4"]
    with NWBHDF5IO(str(pose_nwb), "r", load_namespaces=True) as io:
        series = io.read().acquisition["video"]
        assert series.format == "external"
        assert list(series.starting_frame) == [0]
        assert list(series.dimension) == [640, 480]
        assert series.rate == pytest.approx(FPS)
        assert series.starting_time == 0.0
        assert series.num_samples == NUM_FRAMES


def test_link_external_video_leaves_pose_data_unchanged(pose_nwb: Path, video: Path) -> None:
    """Linking only adds acquisition/video; no existing dataset is changed."""
    before = _datasets(pose_nwb)

    link_external_video(pose_nwb, video, num_frames=NUM_FRAMES, rate=FPS)

    after = _datasets(pose_nwb)
    assert set(before) <= set(after)
    assert all(np.array_equal(before[name], after[name]) for name in before)
    assert all(name.startswith("acquisition/video/") for name in set(after) - set(before))


def test_link_external_video_leaves_no_temporary_file(pose_nwb: Path, video: Path) -> None:
    """The temporary copy used while linking does not outlive the call."""
    link_external_video(pose_nwb, video, num_frames=NUM_FRAMES, rate=FPS)

    assert [p.name for p in pose_nwb.parent.iterdir()] == ["clip.nwb"]


def test_link_external_video_missing_video_leaves_file_untouched(
    pose_nwb: Path, tmp_path: Path
) -> None:
    """A missing video fails before the NWB file is modified."""
    before = pose_nwb.read_bytes()

    with pytest.raises(FileNotFoundError):
        link_external_video(
            pose_nwb, tmp_path / "videos" / "absent.mp4", num_frames=NUM_FRAMES, rate=FPS
        )

    assert pose_nwb.read_bytes() == before
    assert [p.name for p in pose_nwb.parent.iterdir()] == ["clip.nwb"]


def test_link_external_video_twice_fails_and_keeps_first_link(pose_nwb: Path, video: Path) -> None:
    """A second link is refused and the first one is kept intact."""
    link_external_video(pose_nwb, video, num_frames=NUM_FRAMES, rate=FPS)
    linked = pose_nwb.read_bytes()

    with pytest.raises(ValueError, match="already has an acquisition object"):
        link_external_video(pose_nwb, video, num_frames=NUM_FRAMES, rate=FPS)

    assert pose_nwb.read_bytes() == linked
    assert [p.name for p in pose_nwb.parent.iterdir()] == ["clip.nwb"]


def test_link_external_video_requires_pynwb_4(
    monkeypatch: pytest.MonkeyPatch, pose_nwb: Path, video: Path
) -> None:
    """An older pynwb is refused with an error that names the required version."""
    monkeypatch.setattr(nwb_video.pynwb, "__version__", "3.1.3")

    with pytest.raises(ImportError, match=r"pynwb>=4\.0"):
        link_external_video(pose_nwb, video, num_frames=NUM_FRAMES, rate=FPS)
