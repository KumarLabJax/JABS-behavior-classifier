"""jabs io package

TODO: migrate existing io code to this package
"""

from .prediction import PREDICTION_FILE_VERSION, save_predictions

__all__ = [
    "PREDICTION_FILE_VERSION",
    "save_predictions",
]
