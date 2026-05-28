import hashlib
import os
import re
import shutil
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def hide_stderr() -> Generator[int, Any, None]:
    """Context manager to temporarily suppress output to standard error (stderr).

    Redirects all output sent to stderr to os.devnull while the context is active,
    restoring stderr to its original state upon exit.

    Yields:
        int: The file descriptor for stderr.
    """
    fd = sys.stderr.fileno()

    # copy fd before it is overwritten
    with os.fdopen(os.dup(fd), "wb") as copied:
        sys.stderr.flush()

        # open destination
        with open(os.devnull, "wb") as fout:
            os.dup2(fout.fileno(), fd)
        try:
            yield fd
        finally:
            # restore stderr to its previous value
            sys.stderr.flush()
            os.dup2(copied.fileno(), fd)


def hash_file(file: Path):
    """return hash"""
    chunk_size = 8192
    with file.open("rb") as f:
        h = hashlib.blake2b(digest_size=20)
        c = f.read(chunk_size)
        while c:
            h.update(c)
            c = f.read(chunk_size)
    return h.hexdigest()


def get_bool_env_var(var_name, default_value=False) -> bool:
    """Gets a boolean value from an environment variable.

    Args:
        var_name: The name of the environment variable.
        default_value: The default value to return if the variable is
            not set or invalid.

    Returns:
        A boolean value.
    """
    value = os.getenv(var_name)
    if value is None:
        return default_value

    return value.lower() in ("true", "1", "yes", "on", "y", "t")


_POSE_SUFFIX_RE = re.compile(r"_pose_est_v\d+$")


def pose_file_stem(path: str | Path) -> str:
    """Return the base name of a pose file with the ``_pose_est_vN`` suffix removed.

    For example, ``"video_pose_est_v6.h5"`` becomes ``"video"``. If the input does
    not include the ``_pose_est_vN`` suffix, the stem is returned unchanged
    (this allows callers to pass either a pose file path or a video file path
    and get a consistent identifier).

    Args:
        path: A pose file path (or any path-like value) whose stem may include
            the ``_pose_est_vN`` suffix.

    Returns:
        The stem with any trailing ``_pose_est_vN`` suffix stripped.
    """
    return _POSE_SUFFIX_RE.sub("", Path(path).stem)


def copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy a file to ``destination`` so the replacement is atomic.

    The file is copied (via :func:`shutil.copy2`) into a sibling temporary file
    in ``destination``'s parent directory, then renamed into place with
    :meth:`pathlib.Path.replace`. ``destination``'s parent directory is created
    if it does not exist.

    Because the temporary file is created in the same directory as
    ``destination``, the final rename is on the same filesystem and is
    therefore atomic on POSIX and Windows. ``source`` may live on a different
    filesystem (e.g. a ``tempfile.TemporaryDirectory()`` on ``tmpfs``); the
    intermediate copy is what makes cross-filesystem sources safe.

    Readers of ``destination`` never observe a partially written file: they
    see either the previous contents or the new contents.

    Args:
        source: Path to the file whose contents should be installed at
            ``destination``. Metadata is preserved via :func:`shutil.copy2`.
        destination: Final path. Any existing file at this path is replaced.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, tmp_path)
    tmp_path.replace(destination)


def to_safe_name(behavior: str) -> str:
    """Create a version of the given behavior name that is safe to use in filenames.

    Args:
        behavior: string behavior name

    Returns:
        sanitized behavior name

    Raises:
        ValueError: if the behavior name is empty after sanitization
    """
    safe_behavior = re.sub(r"[^\w.-]+", "_", behavior, flags=re.UNICODE)
    safe_behavior = re.sub("_{2,}", "_", safe_behavior)
    safe_behavior = safe_behavior.strip("_")
    if safe_behavior == "":
        raise ValueError("Behavior name is empty after sanitization.")
    return safe_behavior
