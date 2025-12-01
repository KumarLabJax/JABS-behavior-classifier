"""
The `jabs.postprocess` package provides tools for postprocessing behavior predictions.

It includes the the `DurationFilter` class, which adjusts predictions using duration thresholds,
and `HMMFilter` class, which adjusts raw predictions using hidden markov modelling.
"""

from .base_filter import BaseFilter, BehaviorEvents
from .duration_filter import DurationFilter
from .postprocess import Postprocesser, AVAILABLE_FILTERS
