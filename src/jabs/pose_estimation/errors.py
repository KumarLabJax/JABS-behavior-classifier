"""Errors raised when a pose file cannot be opened."""

from pathlib import Path

__all__ = ["PoseOpenError"]


class PoseOpenError(Exception):
    """A pose file could not be opened or converted.

    Wraps the underlying failure (a converter refusal, an unreadable or
    malformed pose file, an undecodable component, an operating system error)
    as its ``__cause__``.

    Attributes:
        path: The pose file the user has.
        reason: Why it could not be opened.
    """

    def __init__(self, path: Path, reason: str) -> None:
        """Record the file and the reason.

        Args:
            path: The pose file the user has.
            reason: Why it could not be opened.
        """
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason

    def __reduce__(self) -> tuple[type["PoseOpenError"], tuple[Path, str]]:
        """Pickle by path and reason, so the error crosses process boundaries."""
        return type(self), (self.path, self.reason)
