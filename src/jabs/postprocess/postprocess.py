import abc
import json
from pathlib import Path

import numpy as np

DEFAULT_POSTPROCESSOR_JSON = Path('postprocess.json')
DEFAULT_POSTPROCESSOR_FOLDER = Path('./filters/')
AVAILABLE_FILTERS = [DurationFilter]

_VERSION = 1

class BehaviorEvents:
    """Structure for interacting with behavioral event data."""

    def __init__(self, starts: np.ndarray, durations: np.ndarray, states: np.ndarray):
        """Constructs an event object.

        Args:
            starts: starting index of events
            durations: duration of events
            states: states of events
        """
        self.validate_rle_data(starts, durations, states)
        self._starts = starts
        self._durations = durations
        self._states = states

    @property
    def starts(self):
        """Integers indicating the frame where an event starts."""
        return self._starts

    @property
    def durations(self):
        """Integers indicating the duration of events."""
        return self._durations

    @property
    def states(self):
        """Values of the state for each event."""
        return self._states

    @classmethod
    def from_vector(cls, vector: np.ndarray):
        """Constructs an event object from vector data.

        Args:
            vector: state vector of behavioral events

        Returns:
            BehaviorEvent object
        """
        return cls(*cls.rle(vector))

    def delete_bouts(self, indices_to_remove):
        """Helper function to delete events from bout data.

        Args:
            indices_to_remove: event indices to delete

        Returns:
            Bouts object that has been modified to interpolate within deleted events

        Notes:
            Interpolation on an odd number will result with the "previous" state getting 1 more frame compared to "next" state
        """
        new_durations = np.copy(self.durations)
        new_starts = np.copy(self.starts)
        new_states = np.copy(self.states)
        if len(indices_to_remove) > 0:
            # Delete backwards so that we don't need to shift indices
            for cur_gap in np.sort(indices_to_remove)[::-1]:
                # Nothing earlier or later to join together, ignore
                if cur_gap == 0 or cur_gap == len(new_durations) - 1:
                    pass
                else:
                    # Delete gaps where the borders match
                    if new_states[cur_gap - 1] == new_states[cur_gap + 1]:
                        # Adjust surrounding data
                        cur_duration = np.sum(new_durations[cur_gap - 1 : cur_gap + 2])
                        new_durations[cur_gap - 1] = cur_duration
                        # Since the border bouts merged, delete the gap and the 2nd bout
                        new_durations = np.delete(new_durations, [cur_gap, cur_gap + 1])
                        new_starts = np.delete(new_starts, [cur_gap, cur_gap + 1])
                        new_states = np.delete(new_states, [cur_gap, cur_gap + 1])
                    # Delete gaps where the borders don't match by dividing the block in half
                    else:
                        # Adjust surrounding data
                        # To remove rounding issues, round down for left, up for right
                        duration_deleted = new_durations[cur_gap]
                        # Previous bout gets longer
                        new_durations[cur_gap - 1] = new_durations[cur_gap - 1] + int(
                            np.floor(duration_deleted / 2)
                        )
                        # Next bout also needs start time adjusted
                        new_durations[cur_gap + 1] = new_durations[cur_gap + 1] + int(
                            np.ceil(duration_deleted / 2)
                        )
                        new_starts[cur_gap + 1] = new_starts[cur_gap + 1] - int(
                            np.ceil(duration_deleted / 2)
                        )
                        # Delete out the gap
                        new_durations = np.delete(new_durations, [cur_gap])
                        new_starts = np.delete(new_starts, [cur_gap])
                        new_states = np.delete(new_states, [cur_gap])
        self._starts = new_starts
        self._durations = new_durations
        self._states = new_states

    @staticmethod
    def rle(inarray: np.ndarray):
        """Run-length encode value data.

        Args:
                inarray: input array of data to RLE

        Returns:
                tuple of (starts, durations, states)
                starts: start indices of events
                durations: duration of events
                states: state value of events
        """
        ia = np.asarray(inarray)
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]
            i = np.append(np.where(y), n - 1)
            z = np.diff(np.append(-1, i))
            p = np.cumsum(np.append(0, z))[:-1]
            return (p, z, ia[i])

    @staticmethod
    def to_vector(starts: np.ndarray, durations: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Converts RLE-encoded data back into a vector.

        Args:
                starts: starting index of events
                durations: duration of events
                states: states of events

        Returns:
                The encoded state vector
        """
        BehaviorEvents.validate_rle_data(starts, durations, states)
        if len(starts) == 0:
            return np.asarray([], dtype=states.dtype)

        ends = starts + durations
        vector = np.zeros(ends[-1], dtype=states.dtype)

        for i in np.arange(len(starts)):
            vector[starts[i] : ends[i]] = states[i]

        return vector

    @staticmethod
    def validate_rle_data(starts: np.ndarray, durations: np.ndarray, states: np.ndarray):
        """Validates that RLE data is formatted properly.

        Args:
            starts: starting index of events
            durations: duration of events
            states: states of events

        Raises:
            ValueError if data is not formatted properly
        """
        # Data must be same shape
        if len(starts) != len(durations) or len(durations) != len(states):
            raise ValueError(
                f"All inputs must be of same length. Recieved vectors of shape {len(starts)}, {len(durations)}, {len(states)}."
            )

        # Starts must be sorted
        if not np.all(starts[:-1] <= starts[1:]):
            raise ValueError("Starts of events are not ascending.")

        # Starts + durations must align with next starts
        ends = starts + durations
        if not np.all(ends[:-1] == starts[1:]):
            raise ValueError("Starts + duration must yield the next start value.")

        # States should not sequentially repeat
        repeated_states = states[1:] == states[:-1]
        if np.any(repeated_states):
            raise ValueError(
                f"States cannot repeat. State(s) {states[np.where(repeated_states)]} were repeated in at the following indices: {np.where(repeated_states)}."
            )


class BaseFilter:
    """A filter that adjusts classifier predictions."""
    def __init__(self):
        self._name = 'BaseFilter'
        self._extension = '.filter'

    @property
    def name(self):
        return self._name

    @property
    def file_ext(self):
        return self._file_ext

    @abc.abstractmethod
    def train(self, label: np.ndarray, group: np.ndarray, frame: np.ndarray):
        """Trains the filter from annotated data.

        Args:
            label: annotated behavioral state data
            group: value indicating group labels belong to
            frame: frame information of labels
        """
        pass

    @abc.abstractmethod
    def filter(self, prob: np.ndarray, state: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        """Filters prediction data.

        Args:
            prob: probability matrix of shape [frame, class]
            state: vector of predicted class

        Returns:
            tuple of
                filtered probability matrix
                filtered state vector
        """
        pass

    @abc.abstractmethod
    def save(self, file: Path):
        """Saves filter settings to file.

        Args:
            file: file to write trained filter model settings
        """
        pass

    @abc.abstractmethod
    def load(self, file: Path):
        """Loads filter settings from file.

        Args:
            file: file to load trained filter model settings
        """
        pass


class DurationFilter(BaseFilter):
    """A filter that adjusts prediction based on duration settings."""

    def __init__(self, kwargs: dict):
        """Initializes a duration filter that does nothing."""
        self._interpolate_duration = kwargs.get("interpolate_duration", 0)
        self._stitch_duration = kwargs.get("stitch_duration", 0)
        self._filter_duration = kwargs.get("filter_duration", 0)

        self._name = 'DurationFilter'
        self._file_ext = '.duration.filter'

    def train(self, label: np.ndarray, group: np.ndarray, frame: np.ndarray):
        """Duration filters are not trained."""
        pass

    def filter(self, prob: np.ndarray, state: np.ndarray) -> tuple(np.ndarray, np.ndarray):
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
        rle_data = BehaviorEvents(state)
        interpolate_to_remove = np.logical_and(
            rle_data.duration < self._interpolate_duration, rle_data.states == -1
        )
        if np.any(interpolate_to_remove):
            rle_data.delete_bouts(np.where(interpolate_to_remove)[0])

        stitch_to_remove = np.logical_and(
            rle_data.duration < self._stitch_duration, rle_data.states == 0
        )
        if np.any(stitch_to_remove):
            rle_data.delete_bouts(np.where(stitch_to_remove)[0])

        filter_to_remove = np.logical_and(
            rle_data.duration < self._filter_duration, rle_data.states == 1
        )
        if np.any(filter_to_remove):
            rle_data.delete_bouts(np.where(filter_to_remove)[0])

    def save(self, file: Path):
        """Saves filter settings to file.

        Args:
            file: file to write trained filter model settings
        """
        payload = {
            "interpolate_duration": self._interpolate_duration,
            "stitch_duration": self._stitch_duration,
            "filter_duration": self._filter_duration,
        }
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

        self._interpolate_duration = payload["interpolate_duration"]
        self._stitch_duration = payload["stitch_duration"]
        self._filter_duration = payload["filter_duration"]


class Postprocesser:
    """Postprocessing object manages applying filters to prediction data."""
    def __init__(self):
        """Initializes a set of filters."""
        self._filter_funcs = {cur_fitler.name: cur_filter for cur_filter in AVAILABLE_FILTERS}
        self._filters = []

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

    def apply_filters(self, prob: np.ndarray, state: np.ndarray) -> tuple(np.ndarray, np.ndarray):
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

        for cur_fitler in self._filters:
            filtered_probs, filtered_states = cur_filter.filter(filtered_probs, filtered_states)

        return filtered_probs, fitlered_states

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
        reutrn cls.load_filters(config)
