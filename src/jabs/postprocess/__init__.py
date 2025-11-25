"""
The `jabs.postprocess` package provides tools for postprocessing behavior predictions.

It includes the the `DurationFilter` class, which adjusts predictions using duration thresholds,
and `HMMFilter` class, which adjusts raw predictions using hidden markov modelling.
"""

from .postprocess import DurationFilter, HMMFilter
