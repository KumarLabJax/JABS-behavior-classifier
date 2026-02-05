import enum

import numpy as np


class ClassLabels(enum.IntEnum):
    """Enumeration for class labels in behavioral event data."""

    NONE = -1
    NOT_BEHAVIOR = 0
    BEHAVIOR = 1


class BehaviorEvents:
    """Structure for interacting with behavioral event data.

    Essentially does a run-length encoding of event data, and provides tools for deleting bouts and interpolation.

    Args:
        starts: starting index of events
        durations: duration of events (in frames)
        states: states of events
    """

    def __init__(self, starts: np.ndarray, durations: np.ndarray, states: np.ndarray):
        self._starts = starts
        self._durations = durations
        self._states = states
        self._validate_rle_data()

    @property
    def starts(self) -> np.ndarray:
        """Integers indicating the frame where an event starts."""
        return self._starts

    @property
    def durations(self) -> np.ndarray:
        """Integers indicating the duration of events."""
        return self._durations

    @property
    def states(self) -> np.ndarray:
        """Values of the state for each event."""
        return self._states

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "BehaviorEvents":
        """Constructs an event object from vector data.

        Args:
            vector: state vector of behavioral events
        Returns:
            BehaviorEvent object
        """
        return cls(*cls._rle(vector))

    def delete_bouts(self, indices_to_remove) -> None:
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
                if cur_gap == 0:
                    # Remove the first bout, merge with the next (use next bout's class)
                    new_durations[1] += new_durations[0]
                    new_starts[1] = new_starts[0]
                    new_durations = np.delete(new_durations, 0)
                    new_starts = np.delete(new_starts, 0)
                    new_states = np.delete(new_states, 0)
                elif cur_gap == len(new_durations) - 1:
                    # Remove the last bout, merge with the previous (use previous bout's class)
                    new_durations[-2] += new_durations[-1]
                    new_durations = np.delete(new_durations, -1)
                    new_starts = np.delete(new_starts, -1)
                    new_states = np.delete(new_states, -1)
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
    def _rle(inarray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run-length encode value data.

        Args:
                inarray: input array of data to RLE
        Returns:
                tuple of (starts, durations, states)
                starts: start indices of events (int)
                durations: duration of events (int)
                states: state value of events (same dtype as input)
        """
        ia = np.asarray(inarray)
        n = len(ia)
        if n == 0:
            # Return three empty arrays: starts and durations as int, states as input dtype
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=ia.dtype),
            )
        else:
            change_points = ia[1:] != ia[:-1]
            run_end_indices = np.append(np.where(change_points), n - 1)
            durations = np.diff(np.append(-1, run_end_indices))
            starts = np.cumsum(np.append(0, durations))[:-1]
            return starts, durations, ia[run_end_indices]

    def to_vector(self) -> np.ndarray:
        """Converts this RLE-encoded data back into a vector.

        Returns:
                The encoded state vector
        """
        self._validate_rle_data()
        if len(self.starts) == 0:
            return np.asarray([], dtype=self.states.dtype)

        ends = self.starts + self.durations
        vector = np.zeros(ends[-1], dtype=self.states.dtype)

        for i in np.arange(len(self.starts)):
            vector[self.starts[i] : ends[i]] = self.states[i]

        return vector

    def _validate_rle_data(self):
        """Validates that RLE data is formatted properly.

        Raises:
            ValueError if data is not formatted properly
        """
        # Data must be same shape
        if len(self.starts) != len(self.durations) or len(self.durations) != len(self.states):
            raise ValueError(
                f"All inputs must be of same length. Received vectors of shape {len(self.starts)}, {len(self.durations)}, {len(self.states)}."
            )

        # Starts must be sorted
        if not np.all(self.starts[:-1] <= self.starts[1:]):
            raise ValueError("Starts of events are not ascending.")

        # Starts + durations must align with next starts
        ends = self.starts + self.durations
        if not np.all(ends[:-1] == self.starts[1:]):
            raise ValueError("Starts + duration must yield the next start value.")

        # States should not sequentially repeat
        repeated_states = self.states[1:] == self.states[:-1]
        if np.any(repeated_states):
            raise ValueError(
                f"States cannot repeat. State(s) {self.states[np.where(repeated_states)]} were repeated in at the following indices: {np.where(repeated_states)}."
            )
