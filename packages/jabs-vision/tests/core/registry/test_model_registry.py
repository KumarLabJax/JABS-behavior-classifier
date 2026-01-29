"""Tests for ModelRegistry."""

import pytest

from jabs.vision.core.registry import MODEL_REGISTRY, ModelRegistry


@pytest.fixture
def registry() -> ModelRegistry:
    """Fixture to provide a fresh ModelRegistry instance."""
    return ModelRegistry()


def test_registry_init(registry: ModelRegistry) -> None:
    """Test that the registry initializes empty."""
    assert registry.list_models() == []
    assert len(registry._registry) == 0


def test_register_decorator(registry: ModelRegistry) -> None:
    """Test registering a model using the decorator."""

    @registry.register("test_model")
    class TestModel:
        pass

    assert "test_model" in registry
    assert registry.get("test_model") is TestModel


@pytest.mark.parametrize("name", ["model_a", "model_b", "custom_name"])
def test_register_custom_names(registry: ModelRegistry, name: str) -> None:
    """Test registering models with different names."""

    @registry.register(name)
    class TestModel:
        pass

    assert name in registry
    assert registry.get(name) is TestModel


def test_register_duplicate_raises_error(registry: ModelRegistry) -> None:
    """Test that registering a duplicate name raises ValueError."""

    @registry.register("duplicate_model")
    class TestModel1:
        pass

    with pytest.raises(ValueError, match="Model 'duplicate_model' is already registered"):

        @registry.register("duplicate_model")
        class TestModel2:
            pass


def test_get_existing_model(registry: ModelRegistry) -> None:
    """Test retrieving an existing model."""

    @registry.register("my_model")
    class MyModel:
        pass

    retrieved_cls = registry.get("my_model")
    assert retrieved_cls is MyModel


def test_get_missing_model_raises_error(registry: ModelRegistry) -> None:
    """Test that getting a missing model raises KeyError with available models."""

    @registry.register("existing_model")
    class Existing:
        pass

    with pytest.raises(KeyError, match="Model 'missing_model' not found"):
        registry.get("missing_model")

    # Check that the error message contains the list of registered models
    try:
        registry.get("missing_model")
    except KeyError as e:
        assert "existing_model" in str(e)


def test_list_models_sorted(registry: ModelRegistry) -> None:
    """Test that list_models returns a sorted list of names."""

    @registry.register("zebra")
    class Zebra:
        pass

    @registry.register("alpha")
    class Alpha:
        pass

    @registry.register("beta")
    class Beta:
        pass

    assert registry.list_models() == ["alpha", "beta", "zebra"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("present", True),
        ("absent", False),
    ],
)
def test_contains(registry: ModelRegistry, name: str, expected: bool) -> None:
    """Test the __contains__ protocol."""
    if expected:

        @registry.register(name)
        class Item:
            pass

    assert (name in registry) is expected


def test_global_registry_instance() -> None:
    """Verify the global MODEL_REGISTRY instance exists and is correct type."""
    assert isinstance(MODEL_REGISTRY, ModelRegistry)
