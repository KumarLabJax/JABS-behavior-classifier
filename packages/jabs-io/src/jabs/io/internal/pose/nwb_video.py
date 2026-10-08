"""Link JABS pose NWB files to the video they were estimated from.

DANDI recognizes a video only when an NWB file points to it through an
``ImageSeries`` in ``acquisition`` whose ``external_file`` names the video. These
helpers read what is needed to build that link and append the ``ImageSeries`` to an
existing pose NWB file without re-converting the pose data.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import NamedTuple

try:
    import pynwb
    from pynwb import NWBHDF5IO
    from pynwb.image import ImageSeries

    _PYNWB_AVAILABLE = True
except ImportError:
    _PYNWB_AVAILABLE = False

from jabs.io.internal.pose.nwb import _IDENTITY_MASK_KEY, _PROCESSING_MODULE_NAME

logger = logging.getLogger(__name__)

# ImageSeries.num_samples, which pynwb requires for an external video timed by a rate,
# first appeared in pynwb 4.0. ndx-pose 0.4.0, which JABS writes pose NWB files with,
# needs pynwb 4.0 as well.
_MIN_PYNWB_MAJOR = 4

VIDEO_SERIES_NAME = "video"
_VIDEO_SERIES_DESCRIPTION = (
    "Video the pose was estimated from. The video is stored outside this NWB file and "
    "is referenced through external_file."
)


class NWBVideoLinkInfo(NamedTuple):
    """What is needed from a pose NWB file to link a video to it.

    Attributes:
        num_frames: Number of frames of pose data in the file.
        rate: Frame rate of the pose data, in frames per second.
        external_files: ``external_file`` entries of every ``ImageSeries`` already in
            ``acquisition``, in file order. Empty when no video is linked yet.
    """

    num_frames: int
    rate: float
    external_files: list[str]


def require_video_link_support() -> None:
    """Raise a helpful error if the NWB dependencies are missing or too old.

    Callers that process many files can use this to fail once, up front, instead of
    once per file.

    Raises:
        ImportError: If pynwb is not installed, or is older than 4.0.
    """
    if not _PYNWB_AVAILABLE:
        raise ImportError(
            "Linking videos to NWB files requires the 'nwb' extra "
            "(install with: pip install 'jabs-io[nwb]')."
        )
    if int(pynwb.__version__.split(".")[0]) < _MIN_PYNWB_MAJOR:
        raise ImportError(
            f"Linking videos to NWB files requires pynwb>={_MIN_PYNWB_MAJOR}.0, "
            f"but pynwb {pynwb.__version__} is installed."
        )


def read_video_link_info(path: Path) -> NWBVideoLinkInfo:
    """Read the pose timing and any existing video links of a pose NWB file.

    Args:
        path: JABS pose NWB file.

    Returns:
        The frame count and rate of the pose data, and the ``external_file`` entries
        of the ``ImageSeries`` already in ``acquisition``.

    Raises:
        ImportError: If pynwb is not installed.
        KeyError: If the file does not hold JABS pose data.
    """
    require_video_link_support()
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        identity_mask = nwbfile.processing[_PROCESSING_MODULE_NAME][_IDENTITY_MASK_KEY]
        external_files = [
            file.decode() if isinstance(file, bytes) else str(file)
            for series in nwbfile.acquisition.values()
            if isinstance(series, ImageSeries) and series.external_file is not None
            for file in series.external_file
        ]
        return NWBVideoLinkInfo(
            num_frames=int(identity_mask.data.shape[0]),
            rate=float(identity_mask.rate),
            external_files=external_files,
        )


def relative_video_path(nwb_path: Path, video_path: Path) -> str:
    """Return the path of a video relative to the directory of an NWB file.

    NWB, ``dandi organize`` and ``nwbinspector`` all resolve a relative
    ``external_file`` against the directory of the NWB file.

    Args:
        nwb_path: NWB file the path will be stored in.
        video_path: Video file to point to.

    Returns:
        POSIX-style relative path from the NWB file's directory to the video.

    Raises:
        ValueError: If no relative path exists, such as on Windows when the two files
            are on different drives.
    """
    relative = os.path.relpath(os.path.abspath(video_path), os.path.abspath(nwb_path.parent))
    return Path(relative).as_posix()


def link_external_video(
    nwb_path: Path,
    video_path: Path,
    *,
    num_frames: int,
    rate: float,
    dimension: tuple[int, int] | None = None,
    series_name: str = VIDEO_SERIES_NAME,
) -> str:
    """Add an ``ImageSeries`` pointing to a video to a pose NWB file, in place.

    The change is made on a temporary copy that is read back and checked before it
    replaces the original, so a failure leaves the original file untouched.

    The video is timed with a start time of 0 and the given rate, like the pose series
    in the file, so the video and the pose share one timeline.

    Args:
        nwb_path: Pose NWB file to modify.
        video_path: Video the pose was estimated from. Must exist.
        num_frames: Number of frames in the video, which is the number of pose frames.
        rate: Frame rate of the video, in frames per second.
        dimension: Optional ``(width, height)`` of the video in pixels.
        series_name: Name of the ``ImageSeries`` in ``acquisition``.

    Returns:
        The ``external_file`` path that was stored, relative to the NWB file's directory.

    Raises:
        ImportError: If pynwb is not installed, or is older than 4.0.
        FileNotFoundError: If the video does not exist.
        ValueError: If ``acquisition`` already holds an object named ``series_name``, or
            the stored path does not resolve back to the video.
    """
    require_video_link_support()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    relative = relative_video_path(nwb_path, video_path)
    temp_path = nwb_path.with_name(f".{nwb_path.name}.linking")
    logger.info("Linking %s to %s as %r", nwb_path, relative, series_name)

    shutil.copyfile(nwb_path, temp_path)
    try:
        with NWBHDF5IO(str(temp_path), mode="a", load_namespaces=True) as io:
            nwbfile = io.read()
            if series_name in nwbfile.acquisition:
                raise ValueError(
                    f"{nwb_path} already has an acquisition object named {series_name!r}"
                )
            nwbfile.add_acquisition(
                ImageSeries(
                    name=series_name,
                    description=_VIDEO_SERIES_DESCRIPTION,
                    unit="n.a.",
                    format="external",
                    external_file=[relative],
                    starting_frame=[0],
                    dimension=list(dimension) if dimension is not None else None,
                    rate=float(rate),
                    starting_time=0.0,
                    num_samples=num_frames,
                )
            )
            io.write(nwbfile)

        if relative not in read_video_link_info(temp_path).external_files:
            raise ValueError(f"Link to {relative} was not found when reading {nwb_path} back")
        if not (nwb_path.parent / relative).samefile(video_path):
            raise ValueError(f"{relative} does not resolve to {video_path} from {nwb_path.parent}")
        os.replace(temp_path, nwb_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return relative
