"""Model registry for jabs-vision.

Provides a centralized registry for model classes with decorator-based registration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class ModelRegistry:
    """Registry for model classes.

    Usage:
        @MODEL_REGISTRY.register("my_model")
        class MyModel:
            ...

        # Later:
        model_cls = MODEL_REGISTRY.get("my_model")
    """

    def __init__(self) -> None:
        self._registry: dict[str, type] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator to register a model class.

        Args:
            name: Unique name for the model.

        Returns:
            Decorator function that registers the class.

        Raises:
            ValueError: If a model with this name is already registered.
        """

        def _register(cls: T) -> T:
            if name in self._registry:
                raise ValueError(f"Model '{name}' is already registered.")
            self._registry[name] = cls  # type: ignore[assignment]
            return cls

        return _register

    def get(self, name: str) -> type:
        """Retrieve a model class from the registry.

        Args:
            name: Name of the registered model.

        Returns:
            The registered model class.

        Raises:
            KeyError: If the model name is not found.
        """
        if name not in self._registry:
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Registered models: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list_models(self) -> list[str]:
        """List all registered model names.

        Returns:
            Sorted list of registered model names.
        """
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a model name is registered."""
        return name in self._registry


MODEL_REGISTRY = ModelRegistry()
