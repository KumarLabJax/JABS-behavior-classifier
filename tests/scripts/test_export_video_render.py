"""``jabs-cli export-video`` renders segmentation from legacy and unified pose files.

Unlike ``test_export_video_cli.py``, nothing here is stubbed: the real pose readers and
the real Qt renderer draw a few frames of a v6 pose file with segmentation, as a legacy
file (with its legacy reader, and converted under ``JABS_FORCE_UNIFIED``) and as its
unified conversion.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from jabs.core.utils import hash_file
from jabs.pose_estimation import conversion_cache, open_pose_file
from jabs.scripts.cli.cli import cli

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    SKIP_UI_TESTS = False
    SKIP_REASON = ""
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(SKIP_UI_TESTS, reason=SKIP_REASON)

# a 3600-frame v6 file with segmentation; its contours lie within 800x800
SEGMENTATION_SOURCE = Path(__file__).parents[1] / "data" / "sample_pose_est_v6.h5"
SIZE = 800
FRAMES = 3

CASES = (
    pytest.param("legacy", marks=pytest.mark.legacy_reader),
    "legacy-forced-unified",
    "unified",
)


@pytest.fixture(scope="session")
def unified_v6(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The unified conversion of ``SEGMENTATION_SOURCE``, converted once per session."""
    work = tmp_path_factory.mktemp("unified-v6")
    legacy = work / "source_pose_est_v6.h5"
    shutil.copyfile(SEGMENTATION_SOURCE, legacy)
    return conversion_cache.ensure_converted(legacy, hash_file(legacy), work)


def _write_video(path: Path) -> Path:
    """Write a short black MJPG video large enough to hold the contours."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (SIZE, SIZE))
    try:
        for _ in range(FRAMES):
            writer.write(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def _read_frames(path: Path) -> list[np.ndarray]:
    """Read every frame of a video."""
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return frames
            frames.append(frame)
    finally:
        capture.release()


@pytest.mark.parametrize("case", CASES)
def test_export_video_draws_segmentation(
    tmp_path: Path, unified_v6: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """The default --segmentation export renders, and its contours change the frames."""
    if case == "unified":
        pose_path = tmp_path / "clip_pose.h5"
        shutil.copyfile(unified_v6, pose_path)
    else:
        pose_path = tmp_path / "clip_pose_est_v6.h5"
        shutil.copyfile(SEGMENTATION_SOURCE, pose_path)
    if case == "legacy-forced-unified":
        monkeypatch.setenv("JABS_FORCE_UNIFIED", "1")
    else:
        monkeypatch.delenv("JABS_FORCE_UNIFIED", raising=False)
    video = _write_video(tmp_path / "clip.avi")

    with open_pose_file(pose_path) as pose:
        assert pose.has_segmentation
        window = pose.segmentation_window(0, 0, FRAMES)
        assert (window.contour_count > 0).all(), "fixture frames should carry contours"

    with_seg = tmp_path / "with.mp4"
    without_seg = tmp_path / "without.mp4"
    runner = CliRunner()
    result = runner.invoke(cli, ["export-video", str(video), "-o", str(with_seg)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        cli, ["export-video", str(video), "-o", str(without_seg), "--no-segmentation"]
    )
    assert result.exit_code == 0, result.output

    drawn = _read_frames(with_seg)
    plain = _read_frames(without_seg)
    assert len(drawn) == len(plain) == FRAMES
    for frame_with, frame_without in zip(drawn, plain, strict=True):
        difference = np.abs(frame_with.astype(int) - frame_without.astype(int))
        assert (difference > 50).any(), "segmentation contours should be drawn"
