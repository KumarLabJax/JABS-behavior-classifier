"""The conversion cache: legacy pose files converted once, opened as ``Pose``.

A legacy ``*_pose_est_v<N>.h5`` file is converted to a unified pose file at
``<cache root>/converted/<video stem>.<legacy hash>-r<converter revision>.h5``
and reused while the legacy file's bytes and the converter revision
(``jabs.io.internal.pose_file.convert.CONVERTER_REVISION``, bumped only when the
converter's output changes) are unchanged.
The cache root is the project's ``cache_dir``, or the per-user cache directory
when there is none. The video directory is never written.
"""

import logging
from pathlib import Path

from platformdirs import user_cache_dir

from jabs.core.utils import hash_file, pose_file_stem
from jabs.io.internal.pose_file import PoseFileError
from jabs.io.internal.pose_file import convert as _convert
from jabs.io.internal.pose_file.convert import LegacyConversionError, convert_legacy

from .discovery import POSE_FILE_GLOBS, is_legacy_pose_file, is_unified_pose_file
from .errors import PoseOpenError
from .pose import Pose

logger = logging.getLogger(__name__)

__all__ = ["cache_root", "converted_path", "ensure_converted", "open_unified"]

_APP_NAME = "jabs"
_CONVERTED_DIR = "converted"

# what opening a pose file raises for a file it cannot read; LookupError and
# ValueError cover a malformed manifest that still passes the schema
_OPEN_ERRORS = (PoseFileError, NotImplementedError, OSError, LookupError, ValueError)

# what the converter raises besides OSError: a refusal, or (from a malformed
# legacy file the inspection let through) a KeyError, IndexError or ValueError
_CONVERT_ERRORS = (LegacyConversionError, LookupError, ValueError)


def _converter_revision() -> int:
    """Return the revision of the converter's output (jabs-io's ``CONVERTER_REVISION``)."""
    return _convert.CONVERTER_REVISION


def _reason(path: Path, error: Exception) -> str:
    """Return an error's message without the ``<path>: `` prefix it may start with."""
    message = str(error)
    return message.removeprefix(f"{path}: ") or type(error).__name__


def cache_root(cache_dir: Path | None) -> Path:
    """Return the directory conversions are cached under.

    Args:
        cache_dir: The project's cache directory, or None outside a project.

    Returns:
        ``cache_dir``, or the per-user cache directory for jabs when it is None.
    """
    if cache_dir is not None:
        return cache_dir
    return Path(user_cache_dir(_APP_NAME))


def converted_path(source: Path, source_hash: str, cache_dir: Path | None) -> Path:
    """Return where the conversion of a legacy pose file is cached.

    Args:
        source: The legacy pose file.
        source_hash: ``jabs.core.utils.hash_file`` of ``source``.
        cache_dir: The project's cache directory, or None for the per-user cache.

    Returns:
        ``<cache root>/converted/<video stem>.<source_hash>-r<converter revision>.h5``.
    """
    name = f"{pose_file_stem(source)}.{source_hash}-r{_converter_revision()}.h5"
    return cache_root(cache_dir) / _CONVERTED_DIR / name


def _remove_other_versions(destination: Path, source: Path, source_hash: str) -> None:
    """Delete conversions of the same file (same stem and hash) by other converter revisions.

    This includes copies keyed by a jabs-io version, as earlier JABS named them.

    A copy that cannot be deleted (Windows blocks deleting an open file) is
    logged and left for a later prune.
    """
    prefix = f"{pose_file_stem(source)}.{source_hash}-"
    for path in destination.parent.iterdir():
        if path == destination or not path.name.startswith(prefix) or path.suffix != ".h5":
            continue
        try:
            path.unlink()
        except OSError as error:
            logger.warning("Could not remove stale conversion %s: %s", path.name, error)
        else:
            logger.debug("Removed stale conversion %s", path.name)


def _discard(path: Path) -> None:
    """Delete a cached conversion, logging (not raising) when it cannot be deleted."""
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        logger.warning("Could not remove conversion %s: %s", path.name, error)


def _check_unchanged(source: Path, source_hash: str, destination: Path) -> None:
    """Discard a fresh conversion when the source no longer has the hash it is keyed by.

    Raises:
        PoseOpenError: If the source changed (or became unreadable) during conversion.
    """
    try:
        unchanged = hash_file(source) == source_hash
    except OSError as error:
        _discard(destination)
        raise PoseOpenError(source, f"cannot read after converting: {error}") from error
    if not unchanged:
        _discard(destination)
        raise PoseOpenError(source, "changed while it was being converted; open it again")


def ensure_converted(source: Path, source_hash: str, cache_dir: Path | None) -> Path:
    """Return the cached conversion of a legacy pose file, converting it if missing.

    Two processes converting the same file at once both succeed: the write is
    atomic, and when the final replace fails because the other process's copy
    is in the way (Windows), that copy is used.

    After a conversion the source is hashed again; if it changed while it was
    being converted, the conversion is discarded rather than kept under a key
    that no longer matches the file, and the call fails. Opening the file again
    converts its new content.

    Args:
        source: The legacy pose file.
        source_hash: ``jabs.core.utils.hash_file`` of ``source``.
        cache_dir: The project's cache directory, or None for the per-user cache.

    Returns:
        The converted unified pose file.

    Raises:
        PoseOpenError: If the converter refuses the file or fails on it (a
            malformed file), it cannot be converted or written, or it changed
            while being converted.
    """
    destination = converted_path(source, source_hash, cache_dir)
    if destination.exists():
        logger.debug("Using cached conversion %s of %s", destination, source)
        return destination
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Converting %s to %s", source, destination)
        convert_legacy(source, destination)
        _check_unchanged(source, source_hash, destination)
    except _CONVERT_ERRORS as error:
        raise PoseOpenError(source, _reason(source, error)) from error
    except OSError as error:
        if not destination.exists():
            raise PoseOpenError(source, f"cannot convert: {error}") from error
        logger.info("Using the conversion of %s another process wrote: %s", source, error)
    _remove_other_versions(destination, source, source_hash)
    return destination


def open_unified(path: Path, cache_dir: Path | None = None) -> Pose:
    """Open a pose file as ``Pose``, converting a legacy file through the cache.

    The user's file is hashed once; the hash keys the conversion and is the
    returned ``Pose.hash``. A cached conversion that cannot be opened (a
    truncated or invalid file) is deleted and converted again, once. A unified
    file is never deleted.

    Args:
        path: A unified ``*_pose.h5`` or legacy ``*_pose_est_v<N>.h5`` file.
        cache_dir: The project's cache directory, or None for the per-user cache.
            Convex hulls are cached here too when it is given.

    Returns:
        The opened ``Pose``; the caller closes it.

    Raises:
        PoseOpenError: If the name follows neither pattern, or the file cannot
            be read, converted or opened.
    """
    legacy = is_legacy_pose_file(path)
    if not legacy and not is_unified_pose_file(path):
        raise PoseOpenError(
            path, f"not a pose file name; expected one of {', '.join(POSE_FILE_GLOBS)}"
        )
    try:
        source_hash = hash_file(path)
    except OSError as error:
        raise PoseOpenError(path, f"cannot read: {error}") from error
    if not legacy:
        try:
            return Pose(path, source_hash=source_hash, cache_dir=cache_dir)
        except _OPEN_ERRORS as error:
            raise PoseOpenError(path, _reason(path, error)) from error

    pose_path = ensure_converted(path, source_hash, cache_dir)
    try:
        return Pose(
            pose_path,
            source_path=path,
            source_hash=source_hash,
            cache_dir=cache_dir,
            converter_revision=_converter_revision(),
        )
    except _OPEN_ERRORS as error:
        # a damaged cached copy (truncated, invalid): convert again, once
        logger.warning("Cannot open conversion %s (%s); converting again", pose_path.name, error)
    _discard(pose_path)
    pose_path = ensure_converted(path, source_hash, cache_dir)
    try:
        return Pose(
            pose_path,
            source_path=path,
            source_hash=source_hash,
            cache_dir=cache_dir,
            converter_revision=_converter_revision(),
        )
    except _OPEN_ERRORS as error:
        _discard(pose_path)
        raise PoseOpenError(path, _reason(pose_path, error)) from error
