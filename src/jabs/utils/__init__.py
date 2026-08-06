"""JABS utilities.

The update-check helpers live in ``jabs-core`` and are re-exported here so that
``from jabs.utils import check_for_update`` keeps working for the GUI.
"""

from jabs.core.utils import check_for_update, is_pypi_install

# a hard coded random seed used for the final training done with all
# training data before saving the classifier
# the choice of random seed is arbitrary
FINAL_TRAIN_SEED = 0xAB3BDB

__all__ = [
    "FINAL_TRAIN_SEED",
    "check_for_update",
    "is_pypi_install",
]
