from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jabs.io.backends.base import StorageBackend


@dataclass
class BackendRegistry:
    """Registry for storage backends.

    The first backend whose ``can_handle`` returns True wins.
    """

    _backends: list[StorageBackend] = field(default_factory=list)

    def register(self, backend: StorageBackend, *, first: bool = False) -> None:
        """Register a backend.

        Args:
            backend: Backend instance to register.
            first: If True, insert at the front for higher priority.
        """
        if first:
            self._backends.insert(0, backend)
        else:
            self._backends.append(backend)

    def unregister(self, backend: StorageBackend) -> None:
        """Remove a backend if present."""
        try:
            self._backends.remove(backend)
        except ValueError:
            return

    def clear(self) -> None:
        """Remove all backends."""
        self._backends.clear()

    def list(self) -> tuple[StorageBackend, ...]:
        """Return registered backends in priority order."""
        return tuple(self._backends)

    def get_backend(self, path: Path) -> StorageBackend:
        """Return the first backend that can handle ``path``."""
        for backend in self._backends:
            if backend.can_handle(path):
                return backend
        raise ValueError(f"No backend found for file: {path}")


BACKEND_REGISTRY = BackendRegistry()
