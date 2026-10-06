"""JABS utilities.

Everything this module exports is defined in ``jabs-core`` and re-exported here
so that ``from jabs.utils import ...`` keeps working for the GUI. Nothing new
should be defined in this module: the canonical home for shared constants is
:mod:`jabs.core.constants`, and for shared helpers :mod:`jabs.core.utils`.
"""

from jabs.core.constants import FINAL_TRAIN_SEED
from jabs.core.utils import check_for_update, is_pypi_install

__all__ = [
    "FINAL_TRAIN_SEED",
    "check_for_update",
    "is_pypi_install",
]
