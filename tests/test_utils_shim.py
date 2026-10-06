"""Tests for the ``jabs.utils`` compatibility shim.

``jabs.utils`` exists only to re-export names that live in ``jabs-core``. These
tests pin that contract so a name cannot quietly acquire a second definition
here and drift away from the canonical one.
"""

import jabs.core.constants
import jabs.core.utils
import jabs.utils


def test_final_train_seed_is_the_core_constant() -> None:
    """The exported training seed must be the one ``jabs-core`` defines.

    The seed is what makes the final (post-cross-validation) fit reproducible,
    so two copies of it that disagree would make two call sites train two
    different classifiers from the same data.
    """
    assert jabs.utils.FINAL_TRAIN_SEED is jabs.core.constants.FINAL_TRAIN_SEED


def test_update_check_helpers_are_the_core_functions() -> None:
    """The exported update-check helpers must be ``jabs-core``'s functions."""
    assert jabs.utils.check_for_update is jabs.core.utils.check_for_update
    assert jabs.utils.is_pypi_install is jabs.core.utils.is_pypi_install
