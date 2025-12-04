from . import BaseFilter, BehaviorEvents
import numpy as np
from pathlib import Path
import json

class DurationFilter(BaseFilter):
    """A filter that adjusts prediction based on duration settings."""

    def __init__(self, kwargs: dict = {}):
        """Initializes a duration filter that does nothing."""
        super().__init__()
        self._name = 'DurationFilter'
        self._file_ext = '.duration.filter'
        self._kwargs = {
            "interpolate_duration": kwargs.get("interpolate_duration", 0),
            "stitch_duration": kwargs.get("stitch_duration", 0),
            "filter_duration": kwargs.get("filter_duration", 0),
        }

    def train(self, label: np.ndarray, group: np.ndarray, frame: np.ndarray):
        """Duration filters are not trained."""
        pass

    def filter(self, prob: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filters prediction data.

        Args:
            prob: probability matrix of shape [frame, class]
            state: vector of predicted class

        Returns:
            tuple of
                filtered probability matrix
                filtered state vector

        Notes:
            Filters are applied in the order of:
                1. Interpolate: removes "no prediction"
                2. Stitch: removes "not behavior"
                3. Filter: removes "behavior"
        """
        rle_data = BehaviorEvents.from_vector(state)
        interpolate_to_remove = np.logical_and(
            rle_data.durations < self._kwargs["interpolate_duration"], rle_data.states == -1
        )
        if np.any(interpolate_to_remove):
            rle_data.delete_bouts(np.where(interpolate_to_remove)[0])

        stitch_to_remove = np.logical_and(
            rle_data.durations < self._kwargs["stitch_duration"], rle_data.states == 0
        )
        if np.any(stitch_to_remove):
            rle_data.delete_bouts(np.where(stitch_to_remove)[0])

        filter_to_remove = np.logical_and(
            rle_data.durations < self._kwargs["filter_duration"], rle_data.states == 1
        )
        if np.any(filter_to_remove):
            rle_data.delete_bouts(np.where(filter_to_remove)[0])
        
        return prob, rle_data.to_vector(rle_data.starts, rle_data.durations, rle_data.states)

    def save(self, file: Path):
        """Saves filter settings to file.

        Args:
            file: file to write trained filter model settings
        """
        payload = self._kwargs
        with open(file, "w") as f:
            json.dumps(payload, f)

    def load(self, file: Path):
        """Loads filter settings from file.

        Args:
            file: file to load trained filter model settings

        Raises:
            KeyError if proper keys are not present in json
        """
        with open(file) as f:
            payload = json.loads(f)

        self._kwargs = payload
