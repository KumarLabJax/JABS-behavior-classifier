"""The conversion cache: legacy pose files converted once, opened as ``Pose``.

A legacy ``*_pose_est_v<N>.h5`` file is converted to a unified pose file at
``<cache root>/converted/<video stem>.<legacy hash>-r<converter revision>.h5``
and reused while the legacy file's bytes and the converter revision
(``jabs.io.internal.pose_file.convert.CONVERTER_REVISION``, bumped only when the
converter's output changes) are unchanged.
Conversions are lenient unless ``$JABS_STRICT_POSE_CONVERSION`` is set to a
true value (``1``, ``true``, ``yes``): an unknown legacy dataset, root group or
static object, a malformed dynamic object, or an ``external_identity_mapping``
with other than one name per identity is left behind and recorded in the
convert entry (``dropped``, and ``extra["waived"]`` with the reason) rather
than make the video unusable, as JABS's legacy readers ignored them. Strict
mode keeps the converter's refusals, and its copies are keyed apart
(``-r<revision>-strict.h5``), so a strict open never uses a lenient copy.
The cache root is the project's ``cache_dir``; without one it is
``$JABS_CACHE_DIR`` when that is set, else the per-user cache directory. When
that directory cannot be written, or has no space or quota left, a one-off
open converts into a temporary directory removed when the ``Pose`` is closed. The video directory is never
written.
"""

import errno
import logging
import os
import re
import shutil
import tempfile
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

__all__ = [
    "STRICT_CONVERSION_ENV",
    "cache_root",
    "converted_path",
    "ensure_converted",
    "is_converted",
    "open_unified",
    "strict_conversion",
]

_APP_NAME = "jabs"
_CONVERTED_DIR = "converted"
# overrides the per-user cache directory for opens without a project
CACHE_DIR_ENV = "JABS_CACHE_DIR"

# set to a true value, conversions keep the converter's refusals
STRICT_CONVERSION_ENV = "JABS_STRICT_POSE_CONVERSION"
_TRUE_VALUES = frozenset({"1", "true", "yes"})
# the name key suffix of a strict conversion's copy
_STRICT_SUFFIX = "-strict"

# the key after ``<stem>.<hash>-`` of a conversion: a converter revision (and
# the strict suffix), or the jabs-io version earlier JABS keyed copies by
# (such as ``0.4.0.dev1``)
_REVISION_KEY = re.compile(r"r(\d+)(?:" + _STRICT_SUFFIX + ")?")
_VERSION_KEY = re.compile(r"\d+(\.\d+)*\S*")

# a full disk, or an exceeded quota (not defined on Windows)
_NO_SPACE_ERRNOS = frozenset(
    code for code in (errno.ENOSPC, getattr(errno, "EDQUOT", None)) if code is not None
)
# how h5py names the errno in an OSError it raises without setting errno
_MESSAGE_ERRNO = re.compile(r"\berrno = (\d+)")

# per-user cache roots already reported as unwritable, so each is reported once
_reported_unwritable: set[Path] = set()

# what opening a pose file raises for a file it cannot read; LookupError and
# ValueError cover a malformed manifest that still passes the schema
_OPEN_ERRORS = (PoseFileError, NotImplementedError, OSError, LookupError, ValueError)

# what the converter raises besides OSError: a refusal, or (from a malformed
# legacy file the inspection let through) a KeyError, IndexError or ValueError
_CONVERT_ERRORS = (LegacyConversionError, LookupError, ValueError)


def _converter_revision() -> int:
    """Return the revision of the converter's output (jabs-io's ``CONVERTER_REVISION``)."""
    return _convert.CONVERTER_REVISION


def strict_conversion() -> bool:
    """Return whether legacy conversions keep the converter's refusals.

    The one place JABS reads ``$JABS_STRICT_POSE_CONVERSION``; it is read on
    every call, so worker processes follow the environment they inherit.

    Returns:
        True when the variable is ``1``, ``true`` or ``yes`` (any case);
        False (lenient) otherwise.
    """
    return os.environ.get(STRICT_CONVERSION_ENV, "").strip().lower() in _TRUE_VALUES


def _strict(strict: bool | None) -> bool:
    """Return ``strict``, or the environment's mode when it is None."""
    return strict_conversion() if strict is None else strict


def _reason(path: Path, error: Exception) -> str:
    """Return an error's message without the ``<path>: `` prefix it may start with."""
    message = str(error)
    return message.removeprefix(f"{path}: ") or type(error).__name__


def cache_root(cache_dir: Path | None) -> Path:
    """Return the directory conversions are cached under.

    Args:
        cache_dir: The project's cache directory, or None outside a project.

    Returns:
        ``cache_dir``; when it is None, ``$JABS_CACHE_DIR`` if set and not
        empty, else the per-user cache directory for jabs.
    """
    if cache_dir is not None:
        return cache_dir
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(user_cache_dir(_APP_NAME))


def converted_path(
    source: Path, source_hash: str, cache_dir: Path | None, *, strict: bool | None = None
) -> Path:
    """Return where the conversion of a legacy pose file is cached.

    Args:
        source: The legacy pose file.
        source_hash: ``jabs.core.utils.hash_file`` of ``source``.
        cache_dir: The project's cache directory, or None for the per-user cache.
        strict: The conversion mode; None for :func:`strict_conversion`.

    Returns:
        ``<cache root>/converted/<video stem>.<source_hash>-r<converter revision>.h5``,
        with ``-strict`` before ``.h5`` for a strict conversion.
    """
    suffix = _STRICT_SUFFIX if _strict(strict) else ""
    name = f"{pose_file_stem(source)}.{source_hash}-r{_converter_revision()}{suffix}.h5"
    return cache_root(cache_dir) / _CONVERTED_DIR / name


def _exists(path: Path) -> bool:
    """Return whether ``path`` exists; False when that cannot be checked.

    Before Python 3.14 ``Path.exists`` raises when a parent directory cannot
    be searched (such as a shared cache's ``converted/`` another user created
    0o700). Treated as missing, the conversion that follows fails with an
    ``OSError`` that is handled.
    """
    try:
        return path.exists()
    except OSError:
        return False


def is_converted(
    source: Path, source_hash: str, cache_dir: Path | None, *, strict: bool | None = None
) -> bool:
    """Return whether the conversion of a legacy pose file is cached.

    Args:
        source: The legacy pose file.
        source_hash: ``jabs.core.utils.hash_file`` of ``source``.
        cache_dir: The project's cache directory, or None for the per-user cache.
        strict: The conversion mode; None for :func:`strict_conversion`.

    Returns:
        Whether :func:`converted_path` exists; False when that cannot be checked.
    """
    return _exists(converted_path(source, source_hash, cache_dir, strict=strict))


def _is_older_conversion(key: str) -> bool:
    """Return whether a conversion's name key (after ``<stem>.<hash>-``) is from an older converter.

    ``r<N>`` (of either mode) is older when ``N`` is below the current revision; a jabs-io
    version, as earlier JABS named copies (``0.4.0``), is always older. Anything
    else, a newer revision included, is left alone: another JABS sharing the
    cache may be using it.
    """
    revision = _REVISION_KEY.fullmatch(key)
    if revision is not None:
        return int(revision.group(1)) < _converter_revision()
    return _VERSION_KEY.fullmatch(key) is not None


def _remove_older_versions(destination: Path, source: Path, source_hash: str) -> None:
    """Delete conversions of the same file (same stem and hash) by older converter revisions.

    This includes copies keyed by a jabs-io version, as earlier JABS named
    them. Copies by a newer revision are kept: a newer JABS sharing the
    project or the per-user cache uses them.

    A copy that cannot be deleted (Windows blocks deleting an open file) is
    logged and left for a later prune.
    """
    prefix = f"{pose_file_stem(source)}.{source_hash}-"
    for path in destination.parent.iterdir():
        if path == destination or not path.name.startswith(prefix) or path.suffix != ".h5":
            continue
        if not _is_older_conversion(path.name[len(prefix) : -len(path.suffix)]):
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


def ensure_converted(
    source: Path, source_hash: str, cache_dir: Path | None, *, strict: bool | None = None
) -> Path:
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
        strict: Keep the converter's refusals (see the module docstring);
            None for :func:`strict_conversion`.

    Returns:
        The converted unified pose file.

    Raises:
        PoseOpenError: If the converter refuses the file or fails on it (a
            malformed file), it cannot be converted or written, or it changed
            while being converted.
    """
    strict = _strict(strict)
    destination = converted_path(source, source_hash, cache_dir, strict=strict)
    if _exists(destination):
        logger.debug("Using cached conversion %s of %s", destination, source)
        return destination
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Converting %s to %s", source, destination)
        convert_legacy(source, destination, strict=strict)
        _check_unchanged(source, source_hash, destination)
    except _CONVERT_ERRORS as error:
        raise PoseOpenError(source, _reason(source, error)) from error
    except OSError as error:
        if not _exists(destination):
            raise PoseOpenError(source, f"cannot convert: {error}") from error
        logger.info("Using the conversion of %s another process wrote: %s", source, error)
    _remove_older_versions(destination, source, source_hash)
    return destination


def open_unified(path: Path, cache_dir: Path | None = None) -> Pose:
    """Open a pose file as ``Pose``, converting a legacy file through the cache.

    The user's file is hashed once; the hash keys the conversion and is the
    returned ``Pose.hash``. A cached conversion that cannot be opened (a
    truncated or invalid file) is deleted and converted again, once. A unified
    file is never deleted. Without ``cache_dir``, a per-user cache that cannot
    be written, or runs out of space or quota while converting, is bypassed:
    the file is converted into a temporary directory the ``Pose`` removes.

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

    # read once, so every step of this open uses one mode
    strict = strict_conversion()
    if cache_dir is None:
        temporary = _temporary_cache_if_unwritable(path, source_hash, strict)
        if temporary is not None:
            return _open_in_temporary_cache(path, source_hash, temporary, strict)

    try:
        return _open_converted(path, source_hash, cache_dir, strict)
    except PoseOpenError as error:
        # a project's cache is never bypassed: its conversions are kept
        if cache_dir is not None or not _out_of_space(error.__cause__):
            raise
        logger.warning(
            "No space left in the pose conversion cache %s (%s); converting into a "
            "temporary directory removed after use. Free space, or set %s to a directory "
            "with room, to keep conversions between runs.",
            cache_root(None),
            error.__cause__,
            CACHE_DIR_ENV,
        )
    return _open_in_temporary_cache(
        path, source_hash, Path(tempfile.mkdtemp(prefix="jabs-pose-")), strict
    )


def _out_of_space(error: BaseException | None) -> bool:
    """Return whether ``error`` is a disk-full or quota-exceeded ``OSError``.

    h5py often raises an ``OSError`` without ``errno`` set, naming it in the
    message (``errno = 28``) instead, so both are checked.
    """
    if not isinstance(error, OSError):
        return False
    if error.errno in _NO_SPACE_ERRNOS:
        return True
    named = _MESSAGE_ERRNO.search(str(error))
    return named is not None and int(named.group(1)) in _NO_SPACE_ERRNOS


def _open_converted(path: Path, source_hash: str, cache_dir: Path | None, strict: bool) -> Pose:
    """Open a legacy file's cached conversion, converting it if missing or damaged.

    Raises:
        PoseOpenError: If the file cannot be converted or its conversion opened.
    """
    pose_path = ensure_converted(path, source_hash, cache_dir, strict=strict)
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
    pose_path = ensure_converted(path, source_hash, cache_dir, strict=strict)
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


def _can_write(directory: Path) -> bool:
    """Return whether files can be created in ``directory``, creating it if needed."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory):
            pass
    except OSError:
        return False
    return True


def _temporary_cache_if_unwritable(source: Path, source_hash: str, strict: bool) -> Path | None:
    """Return a new temporary cache root when the per-user cache cannot take a conversion.

    An existing conversion in the per-user cache is used even when the cache
    is read-only, so None is returned then too. The first time a cache root is
    found unwritable a WARNING is logged.

    Returns:
        A new temporary directory the caller owns, or None to use the per-user cache.
    """
    destination = converted_path(source, source_hash, None, strict=strict)
    if _exists(destination) or _can_write(destination.parent):
        return None
    root = cache_root(None)
    if root not in _reported_unwritable:
        _reported_unwritable.add(root)
        logger.warning(
            "Cannot write the pose conversion cache %s; converting into a temporary "
            "directory removed after use. Set %s to a writable directory to keep "
            "conversions between runs.",
            root,
            CACHE_DIR_ENV,
        )
    return Path(tempfile.mkdtemp(prefix="jabs-pose-"))


def _open_in_temporary_cache(path: Path, source_hash: str, temporary: Path, strict: bool) -> Pose:
    """Convert a legacy file into ``temporary`` and open it; the Pose removes it on close.

    Raises:
        PoseOpenError: If the file cannot be converted or opened; ``temporary``
            is removed.
    """
    try:
        pose_path = ensure_converted(path, source_hash, temporary, strict=strict)
        try:
            return Pose(
                pose_path,
                source_path=path,
                source_hash=source_hash,
                converter_revision=_converter_revision(),
                cleanup_dir=temporary,
            )
        except _OPEN_ERRORS as error:
            raise PoseOpenError(path, _reason(pose_path, error)) from error
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
