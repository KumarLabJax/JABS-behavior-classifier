from dataclasses import dataclass


@dataclass(frozen=True)
class VideoInfo:
    """Metadata about the input video.

    Attributes:
        path: Path to the input video.
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        fps: Frames per second if known.
        frame_count: Total number of frames if known.
    """

    path: str
    width: int
    height: int
    fps: float | None = None
    frame_count: int | None = None
