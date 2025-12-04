import abc
import json
from pathlib import Path

import numpy as np

from . import DurationFilter, HMMFilter, BaseFilter

DEFAULT_POSTPROCESSOR_JSON = Path('postprocess.json')
DEFAULT_POSTPROCESSOR_FOLDER = Path('./filters/')
AVAILABLE_FILTERS = [DurationFilter, HMMFilter]

_VERSION = 1

class Postprocesser:
    """Postprocessing object manages applying filters to prediction data."""
    def __init__(self, filter_defs: list = []):
        """Initializes a set of filters.
        
        Args:
            filter_defs: a list of dicts that contain filter_name:kwarg pairs
        """
        self._filter_funcs = {cur_filter.name: cur_filter for cur_filter in AVAILABLE_FILTERS}
        self._filters = []
        for cur_filter in filter_defs:
            self.add_filter(self._filter_funcs[cur_filter](filter_kwargs))
        self._is_trained = False

    def get_filter_config(self):
        """Gets the filter configuration for changing settings.

        Returns:
            A dict with the following structure:
            {
                name: filter name
                kwargs: filter_kwargs
            }
        """
        return_list = []
        for cur_filter in self._filters:
            return_list.append({
                'name': cur_filter.get_filter_name(),
                'kwargs': cur_filter.get_kwargs(),
            })
        return return_list

    def add_filter(self, new_filter: BaseFilter):
        """Adds a filter to the postprocessor.

        Args:
            new_filter: New filter to add
        """
        self._filters.append(new_filter)

    def train_filters(self, labels: np.ndarray, groups: np.ndarray, frames: np.ndarray):
        """Trains all filters on labeled data.

        Args:
            labels: vector containing labeled states
            groups: vector containing group information
            frames: vector containing frame within group
        """
        for cur_filter in self._filters:
            cur_filter.train(labels, groups, frames)

    def apply_filters(self, prob: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applies the postprocessing filters to predictions.

        Args:
            prob: probability matrix of shape [frame, class]
            state: vector of predicted class

        Returns:
            tuple of
                filtered probability matrix
                filtered state vector
        """
        filtered_probs = np.copy(prob)
        filtered_states = np.copy(state)

        for cur_filter in self._filters:
            filtered_probs, filtered_states = cur_filter.filter(filtered_probs, filtered_states)

        return filtered_probs, filtered_states

    def save_filters(self, config: Path|None = None, folder: Path|None = None):
        """Saves postprocessor configuration to files.

        Args:
            config: json file describing the postprocessor
            folder: folder where child filters are written
        """
        if config is None:
            config = DEFAULT_POSTPROCESSOR_JSON
        if folder is None:
            folder = DEFAULT_POSTPROCESSOR_FOLDER

        # Write out each of the filters
        written_filters = []
        for filter_idx, cur_filter in enumerate(self._filters):
            out_file = folder + Path(str(filter_idx)).with_suffix(cur_filter.file_ext)
            cur_filter.save(out_file)
            written_filters.append({
                "filter_type": cur_filter.name,
                "file": out_file
            })

        # Write out the postprocessing config
        payload = {
            'version': _VERSION,
            'filters': written_filters,
        }
        with open(config, 'w') as f:
            json.dumps(f, payload)

    def load_filters(self, config: Path):
        """Loads a postprocessor from file.

        Args:
            config: json describing the postprocessor
        """
        with open(config) as f:
            payload = json.loads(f)

        if payload["version"] != _VERSION:
            raise ValueError(f"Trying to load postprocessing filters from incompatible runtime version. Runtime version: {_VERSION}, config version: {payload["version"]}")

        self._filters = []
        for cur_filter in payload["filters"]:
            new_filter = self._filter_funcs[cur_filter["filter_type"]]
            new_filter.load(cur_filter["file"])
            self._filters.append(new_filter)

    @classmethod
    def from_config(cls, config: Path):
        """Creates a postprocessor from file.

        Args:
            config: json describign the postprocessor
        """
        postprocessor = cls()
        return cls.load_filters(config)
