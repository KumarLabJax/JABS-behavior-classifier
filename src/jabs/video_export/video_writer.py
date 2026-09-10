"""Write a copy of a video with the JABS pose overlay burned into every frame."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from jabs.video_reader import VideoReader

from .frame_renderer import render_overlay_frame

if TYPE_CHECKING:
    from jabs.pose_estimation import PoseEstimation

logger = logging.getLogger(__name__)

# mp4v is the safe default: H.264 ("avc1") is frequently absent from
# opencv-python-headless builds on Linux, where it fails at writer-open time.
DEFAULT_CODEC = "mp4v"


class VideoExportError(Exception):
    """Raised when an overlay video cannot be written."""


def _reject_writing_over_source(video_path: Path, output_path: Path) -> None:
    """Refuse an output that is the same file as the source.

    Opening the writer truncates the destination, so exporting onto the source
    destroys the very video being read. ``os.path.samefile`` is the check that
    actually holds: comparing resolved paths misses a case-differing name on the
    case-insensitive filesystems of macOS and Windows, and misses hard links
    everywhere. Falls back to a resolved-path comparison only when the output does
    not exist yet, where there is no inode to compare.

    Raises:
        VideoExportError: If the two paths refer to the same file.
    """
    if output_path.exists():
        try:
            same = os.path.samefile(video_path, output_path)
        except OSError:  # pragma: no cover - e.g. the source vanished mid-call
            same = False
    else:
        same = output_path.resolve() == video_path.resolve()

    if same:
        raise VideoExportError(
            f"Output path is the same file as the source video ({video_path}). "
            f"Choose a different output path."
        )


def export_overlay_video(
    video_path: Path,
    output_path: Path,
    pose_est: PoseEstimation,
    *,
    draw_segmentation: bool = True,
    codec: str = DEFAULT_CODEC,
    progress_callback: Callable[[int, int], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> int:
    """Write a copy of a video with the pose overlay burned in.

    Args:
        video_path: Source video to read.
        output_path: Destination video to write. Overwritten if it exists.
        pose_est: Pose estimation for ``video_path``.
        draw_segmentation: Whether to include segmentation contours.
        codec: FourCC codec string passed to ``cv2.VideoWriter``.
        progress_callback: Called after each frame with
            ``(frames_written, total_frames)``.
        should_continue: Polled before each frame; returning False stops the
            export and deletes the partial output, rather than leaving an
            unplayable file behind.

    Returns:
        Number of frames written. Fewer than the video's frame count means the
        export was cancelled.

    Raises:
        VideoExportError: If the output would overwrite the source, the source
            cannot be read, or the writer cannot be opened with the requested
            codec.
    """
    _reject_writing_over_source(video_path, output_path)

    if len(codec) != 4:
        # cv2.VideoWriter_fourcc() takes exactly four characters and raises
        # TypeError otherwise, which would escape as something other than the
        # VideoExportError this function documents.
        raise VideoExportError(
            f"Codec must be a four-character FourCC code, got {codec!r}. "
            f"Examples: 'mp4v', 'avc1', 'MJPG'."
        )

    try:
        reader = VideoReader(video_path)
    except (OSError, ValueError) as e:
        # VideoReader signals an unreadable file with OSError and bad metadata with
        # ValueError; callers of this module only know about VideoExportError.
        raise VideoExportError(f"Could not read {video_path}: {e}") from e

    with reader:
        fps = reader.fps
        if fps <= 0:
            # Reported as a bad frame rate rather than letting the writer fail to
            # open, which would wrongly blame the codec.
            raise VideoExportError(
                f"{video_path} reports a frame rate of {fps}; cannot write a video."
            )

        # The container's frame count and dimensions are estimates for many formats,
        # so neither is trusted: the writer is sized from a real decoded frame, and
        # the loop runs to end-of-stream rather than to a frame count.
        # `reported_frames` is kept only as the denominator for progress.
        reported_frames = reader.num_frames
        first_frame = reader.load_next_frame()["data"]
        if first_frame is None:
            raise VideoExportError(f"{video_path} contains no decodable frames.")
        height, width = first_frame.shape[:2]

        # Frames past the end of the pose data are written without an overlay rather
        # than aborting: a real pose object indexes frame-backed arrays, so asking it
        # for a frame it does not have raises IndexError mid-export.
        #
        # A JABS project cannot hold a mismatch - jabs-init runs the check with
        # enable_video_check=True, which hard-fails - so for the GUI this is
        # defensive. `jabs-cli export-video` is the case that needs it: it takes a
        # loose video and whatever pose file sits beside it, with no project and so
        # no validation that the two agree on length.
        overlay_frames = pose_est.num_frames
        if reported_frames != overlay_frames:
            logger.warning(
                "Frame count mismatch for %s: container reports %d frames, pose has "
                "%d; any frames beyond the pose data are written without an overlay",
                video_path.name,
                reported_frames,
                overlay_frames,
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise VideoExportError(f"Could not create {output_path.parent}: {e}") from e

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if not writer.isOpened():
            raise VideoExportError(
                f"Could not open a video writer for {output_path} using codec {codec!r}. "
                f"This codec may not be available in this OpenCV build."
            )

        frames_written = 0
        cancelled = False
        completed = False
        try:
            frame_data = first_frame
            while frame_data is not None:
                if should_continue is not None and not should_continue():
                    cancelled = True
                    break

                if frame_data.shape[:2] != (height, width):
                    # cv2 silently discards a frame whose size differs from the
                    # writer's, which would yield an empty file reported as success.
                    raise VideoExportError(
                        f"{video_path} changes frame size partway through "
                        f"({width}x{height} to "
                        f"{frame_data.shape[1]}x{frame_data.shape[0]}); cannot export."
                    )

                if frames_written < overlay_frames:
                    output_frame = render_overlay_frame(
                        frame_data,
                        pose_est,
                        frames_written,
                        draw_segmentation=draw_segmentation,
                    )
                else:
                    output_frame = frame_data

                writer.write(output_frame)
                frames_written += 1
                if progress_callback is not None:
                    progress_callback(frames_written, reported_frames)

                frame_data = reader.load_next_frame()["data"]
            completed = True
        finally:
            writer.release()
            # Delete only on cancellation or an exception. A stream that ends before
            # the container's reported count is complete, not truncated, so the frame
            # count must not be used to decide this.
            if cancelled or not completed:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError as unlink_error:
                    # Never let cleanup replace the exception that caused it.
                    logger.warning(
                        "Could not remove partial export %s: %s", output_path, unlink_error
                    )
                else:
                    logger.info(
                        "Removed partial overlay video after %d frame(s) (%s)",
                        frames_written,
                        "cancelled" if cancelled else "export failed",
                    )

    if frames_written != reported_frames:
        logger.info(
            "%s: wrote %d frames; its container reported %d",
            video_path.name,
            frames_written,
            reported_frames,
        )
    return frames_written
