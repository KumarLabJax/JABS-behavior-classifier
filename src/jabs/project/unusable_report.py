"""User-facing summaries of videos whose pose file cannot be used.

An unusable video stays in its project but is left out of training and
classification. These builders turn the project's unusable videos into the one
summary the GUI shows (in a dialog) and the CLI logs (as a WARNING), so the
user is told rather than finding out from the totals. They are plain functions
so they can be tested without Qt.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import NamedTuple

__all__ = ["UnusableNotice", "skipped_training_notice", "unusable_videos_notice"]


class UnusableNotice(NamedTuple):
    """A summary for the user: a short message and the per-video details.

    Attributes:
        message: One or two sentences saying what happened.
        details: One line per video.
    """

    message: str
    details: str

    def as_text(self) -> str:
        """Return the message and details as one block of text, for a log."""
        return f"{self.message}\n{self.details}"


def _videos(count: int) -> str:
    """Return ``"1 video"`` or ``"<count> videos"``."""
    return "1 video" if count == 1 else f"{count} videos"


def unusable_videos_notice(
    unusable: Mapping[str, str], annotated: Collection[str]
) -> UnusableNotice | None:
    """Summarize the unusable videos of a project, flagging those with labels.

    Args:
        unusable: Unusable videos mapped to the reason their pose file cannot be used.
        annotated: Videos that have annotations.

    Returns:
        The notice, or None when no video is unusable.
    """
    if not unusable:
        return None
    labeled = [video for video in unusable if video in annotated]
    message = (
        f"{_videos(len(unusable))} cannot be used because the pose file cannot be "
        "opened. They stay in the project but are left out of training and classification."
    )
    if labeled:
        message += f" {_videos(len(labeled))} with labels will not be trained on."
    lines = [
        f"{video}{' (labeled)' if video in annotated else ''}: {reason}"
        for video, reason in sorted(unusable.items())
    ]
    return UnusableNotice(message, "\n".join(lines))


def skipped_training_notice(
    skipped: Sequence[str], unusable: Mapping[str, str]
) -> UnusableNotice | None:
    """Summarize labeled videos left out of a training run because their pose is unusable.

    Args:
        skipped: Labeled videos that were not trained on.
        unusable: Unusable videos mapped to the reason.

    Returns:
        The notice, or None when no labeled video was skipped.
    """
    if not skipped:
        return None
    message = (
        f"{_videos(len(skipped))} with labels for this training run "
        f"{'was' if len(skipped) == 1 else 'were'} left out because the pose file "
        "cannot be opened. The classifier was trained without their labels."
    )
    lines = [f"{video}: {unusable.get(video, 'unusable')}" for video in sorted(skipped)]
    return UnusableNotice(message, "\n".join(lines))
