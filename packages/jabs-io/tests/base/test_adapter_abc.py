"""Tests for the Adapter abstract base class."""

import pytest

from jabs.io.base import Adapter


def test_adapter_cannot_be_instantiated():
    """Adapter is abstract and should not be directly instantiable."""
    with pytest.raises(TypeError):
        Adapter()


def test_subclass_must_implement_all_abstract_methods():
    """A subclass that omits any abstract method should not be instantiable."""

    class IncompleteAdapter(Adapter):
        @classmethod
        def can_handle(cls, data_type):
            return True

        # Missing: encode, decode, write, read

    with pytest.raises(TypeError):
        IncompleteAdapter()
