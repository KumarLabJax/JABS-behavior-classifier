from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class FormatAdapter(ABC, Generic[T]):
    """Abstract adapter for external formats."""

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Check if this adapter handles the given file."""
        pass

    @abstractmethod
    def read(self, path: Path) -> T:
        """Convert external format to canonical representation."""
        pass

    @abstractmethod
    def write(self, data: T, path: Path) -> None:
        """Convert canonical representation to external format."""
