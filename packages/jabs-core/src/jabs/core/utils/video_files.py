"""Rules for deciding which files in a directory are JABS project videos.

A project directory is scanned for videos in several places that must agree:
the project's ``VideoManager``, the check deciding whether a directory looks
like a project at all, and the CLI tools that remap labels onto an updated pose
set. Each of those goes on to look for the pose file paired with a video, so a
file wrongly counted as a video does not simply go unused - it sends a pose
lookup after a file that is not there or is not readable. Keeping the rule here
is what stops the scans from drifting apart about which files are videos.
"""

from __future__ import annotations

from pathlib import Path

# Filename suffixes JABS recognizes as video files.
VIDEO_FILE_EXTENSIONS: frozenset[str] = frozenset({".avi", ".mp4"})


def find_video_files(directory: Path) -> list[str]:
    """Return the names of the video files directly inside a directory.

    Dotfiles are skipped. In particular this excludes macOS AppleDouble sidecar
    files (``._<name>``) that the OS creates when writing to non-APFS/HFS+
    volumes (e.g. exFAT/NTFS external drives). Those are not real videos, and
    ``pathlib.Path.glob`` matches them (unlike the shell); their companion
    ``._..._pose_est_v*.h5`` sidecars are not valid HDF5, so a scan that treated
    one as a video would fail on the pose file it then went looking for.

    Args:
        directory: Directory to scan. It is not searched recursively.

    Returns:
        Video filenames without their directory, sorted by name. Empty when the
        directory holds no videos.
    """
    return sorted(
        f.name
        for f in directory.glob("*")
        if f.suffix in VIDEO_FILE_EXTENSIONS and not f.name.startswith(".")
    )
