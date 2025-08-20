"""JABS utilities"""

from .utilities import get_bool_env_var, hash_file, hide_stderr

# a hard coded random seed used for the final training done with all
# training data before saving the classifier
# the choice of random seed is arbitrary
FINAL_TRAIN_SEED = 0xAB3BDB

__all__ = [
    "FINAL_TRAIN_SEED",
    "get_bool_env_var",
    "hash_file",
    "hide_stderr",
]
