"""Finding a video's pose file among the unified and legacy file names."""

import re
from pathlib import Path

__all__ = [
    "LEGACY_VERSIONS",
    "POSE_FILE_GLOBS",
    "find_pose_file",
    "is_legacy_pose_file",
    "is_pose_file",
    "is_unified_pose_file",
]

# The two accepted pose file name patterns: unified, then legacy.
POSE_FILE_GLOBS = ("*_pose.h5", "*_pose_est_v*.h5")

# Legacy pose format major versions JABS opens, newest first.
LEGACY_VERSIONS = tuple(range(8, 1, -1))

_UNIFIED_SUFFIX = "_pose.h5"
_LEGACY_NAME_RE = re.compile(r"_pose_est_v(\d+)\.h5$")


def _is_dotfile(path: Path) -> bool:
    """Return True for a hidden file, including macOS AppleDouble ``._*`` files."""
    return path.name.startswith(".")


def is_unified_pose_file(path: Path) -> bool:
    """Return True if ``path`` is named like a unified pose file, ``*_pose.h5``.

    Args:
        path: Any path; only its name is inspected.

    Returns:
        Whether the name is a unified pose file name and not a dotfile.
    """
    return not _is_dotfile(path) and path.name.endswith(_UNIFIED_SUFFIX)


def is_legacy_pose_file(path: Path) -> bool:
    """Return True if ``path`` is named like a legacy pose file JABS opens.

    That is ``*_pose_est_v<N>.h5`` with ``N`` in 2..8.

    Args:
        path: Any path; only its name is inspected.

    Returns:
        Whether the name is a supported legacy pose file name and not a dotfile.
    """
    if _is_dotfile(path):
        return False
    match = _LEGACY_NAME_RE.search(path.name)
    return match is not None and int(match.group(1)) in LEGACY_VERSIONS


def is_pose_file(path: Path) -> bool:
    """Return True if ``path`` is named like a pose file JABS opens.

    Unified ``*_pose.h5`` files and legacy ``*_pose_est_v<N>.h5`` files, ``N``
    in 2..8. Dotfiles (``._*`` AppleDouble files) never are. The file's
    contents are not inspected.

    Args:
        path: Any path; only its name is inspected.

    Returns:
        Whether the name is a pose file name.
    """
    return is_unified_pose_file(path) or is_legacy_pose_file(path)


def find_pose_file(video_path: Path, pose_dir: Path | None = None) -> Path | None:
    """Return the pose file of a video, or None when it has none.

    The unified ``<video stem>_pose.h5`` wins over legacy files; among legacy
    ``<video stem>_pose_est_v<N>.h5`` files the highest ``N`` wins. Names are
    built from the video's stem rather than globbed, so another video whose
    name extends this one's is never matched.

    Args:
        video_path: The video file.
        pose_dir: Directory to search; defaults to the video's directory.

    Returns:
        The pose file's path, or None.
    """
    search_dir = pose_dir if pose_dir is not None else video_path.parent
    stem = video_path.stem
    candidates = [search_dir / f"{stem}_pose.h5"]
    candidates += [search_dir / f"{stem}_pose_est_v{v}.h5" for v in LEGACY_VERSIONS]
    for candidate in candidates:
        if is_pose_file(candidate) and candidate.is_file():
            return candidate
    return None
