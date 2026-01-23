import hashlib
import os
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
