from . import BaseFilter, BehaviorEvents
import numpy as np
from scipy import sparse
from dynamax.hidden_markov_model import CategoricalHMM
from jabs.utils import FINAL_TRAIN_SEED
from pathlib import Path
import json
import os
import jax.numpy as jnp

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

class HMMFilter(BaseFilter):
    """A filter that adjusts prediction based on duration settings."""

    def __init__(self, kwargs: dict = {}):
        """Initializes a duration filter that does nothing."""
        super().__init__()
        self._name = 'HMMFilter'
        self._file_ext = '.hmm.json'
        self._kwargs = {
            'class_list': ['not-behavior', 'behavior'],
            'allowed_unlabeled_frames': 30,
            'random_seed': FINAL_TRAIN_SEED,
        }
        self._model = None
        self._model_params = None
        self._is_trained = False

    def _init_model(self, num_classes: int, random_seed: int | None = None):
        """Initializes an untrained model.

        Args:
            num_classes: number of classes in the HMM
            random_seed: optional random seed used in initial distribution
        """
        # params are num_states, num_emissions, num_classes
        self._model = CategoricalHMM(num_classes, 1, num_classes)

        self._is_trained = False

    @staticmethod
    def _split_by_unlabeled(state_vector: np.ndarray, min_duration: int = 0, unlabeled_val: int = -1):
        """Splits a state vector into sequences by long missing labels.

        Args:
            state_vector: behavior state vector
            min_duration: duration threshold of unlabeled state to split
            unlabeled_val: state value to consider unlabeled

        Returns:
            tuple of (starts, sequence)
            starts: list of starting frame for each sequence
            sequence: list of sequence vectors
        """
        start_frames = []
        sequences = []
        event_data = BehaviorEvents.from_vector(state_vector)
        split_points = np.where(np.logical_and(event_data.states == -1, event_data.durations > min_duration))
        if len(split_points[0]) == 0:
            start_frames.append(0)
            sequences.append(state_vector)
        else:
            last_split = 0
            for cur_split in split_points[0]:
                tmp_state_vector = BehaviorEvents.to_vector(event_data.starts[last_split:cur_split], event_data.durations[last_split:cur_split], event_data.states[last_split:cur_split])
                start_frames.append(event_data.starts[last_split])
                sequences.append(tmp_state_vector)
                # Indexing needs to skip the "no prediction" state, so add 1
                last_split = cur_split + 1
            # Handle the last block
            tmp_state_vector = BehaviorEvents.to_vector(event_data.starts[last_split:], event_data.durations[last_split:], event_data.states[last_split:])
            start_frames.append(event_data.starts[last_split])
            sequences.append(tmp_state_vector)
        return start_frames, sequences

    def labels_to_sequences(self, label: np.ndarray, group: np.ndarray, frame: np.ndarray, max_unlabeled: int):
        """Converts label data to sequences to train on.

        Args:
            label: annotated behavioral state data
            group: value indicating group labels belong to
            frame: frame information of labels
            max_unlabeled: maximum number of unlabeled frames before a split occurs
        
        Returns:
            list of sequences where unlabeled frames have been removed
        """
        sequences = []
        for grp in np.unique(group):
            # Calculate the original state vector with potential unlabeled gaps
            state_vector = label[group == grp]
            frame_vector = frame[group == grp]
            filled_state_vector = np.full(max(frame_vector) + 1, -1)
            filled_state_vector[frame_vector] = state_vector

            # Split on unlabeled gaps larger than allowed unlabeled frames
            _, new_sequences = self._split_by_unlabeled(filled_state_vector, max_unlabeled, -1)
            for cur_sequence in new_sequences:
                # Also remove any remaining missing 
                sequences.append(cur_sequence[cur_sequence != -1])
        
        return sequences

    @staticmethod
    def get_transition_prob(sequences: list[np.ndarray], n_classes: int | None = None):
        """Obtains transition probabilities from a list of sequences.

        Args:
            sequences: list of 1d observation sequences
            n_classes: number of classes

        Returns:
            transition probability matrix of shape [n_classes, n_classes]
        """
        lengths = [len(x) for x in sequences]
        sequence_vec = np.concatenate(sequences)

        if n_classes is None:
            n_classes = np.max(sequence_vec) + 1

        transitions = np.concat([[sequence_vec], [np.roll(sequence_vec, 1)]])
        # Remove invalid transitions (first event of each sequence)
        remove_idxs = np.concatenate([[0], np.cumsum(np.asarray(lengths) - 1)[:-1]])
        transitions = np.delete(transitions, remove_idxs, axis=1)

        from_to, count = np.unique(transitions.T, axis=0, return_counts=True)
        transition_matrix = sparse.csr_matrix((count, from_to.T), shape=(n_classes, n_classes)).toarray()

        return transition_matrix/transition_matrix.sum(axis=1, keepdims=True)


    def train(self, label: np.ndarray, group: np.ndarray, frame: np.ndarray):
        """Trains a hidden markov model.

        Args:
            label: annotated behavioral state data
            group: value indicating group labels belong to
            frame: frame information of labels

        Todo:
            There should be more protection in this function for detecting/removing poor training data.
            Future things to check for:
                * No transitions between states (e.g. no off-diagonals in transition matrix)
                * Poor transition estimates (e.g. low/few off-diagonal)
                * Segments that one may want to exclude from training (e.g. short labels to "correct incorrect predictions")
        """
        n_classes = len(self._kwargs['class_list'])
        self._init_model(n_classes, self._kwargs['random_seed'])

        # Convert labels into a list of sequences
        sequences = self.labels_to_sequences(label, group, frame, self._kwargs["allowed_unlabeled_frames"])
        _, state_counts = np.unique(np.concatenate(sequences), return_counts=True)
        initial_probs = state_counts/state_counts.sum()
        transition_matrix = self.get_transition_prob(sequences, n_classes)
        # Initialize emission probabilities to transition matrix, because it's an okay estimate of our auto-regressive model
        emission_probs = (transition_matrix/transition_matrix.sum(axis=1, keepdims=True)).reshape(self._model.num_states, self._model.emission_dim, self._model.emission_component.num_classes)

        params, properties = self._model.initialize(
            initial_probs=jnp.array(initial_probs),
            transition_matrix=jnp.array(transition_matrix),
            emission_probs=jnp.array(emission_probs),
            )
        self._model_params = params

        self._is_trained = True

    def filter(self, prob: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filters prediction data.

        Args:
            prob: probability matrix of shape [frame, class]
            state: vector of predicted class

        Returns:
            tuple of
                filtered probability matrix
                filtered state vector
        """
        if not self._is_trained:
            raise RuntimeError("Filter not yet trained. Please call .train method before .filter method.")

        new_probs = np.full_like(prob, 0)
        new_states = np.full_like(state, -1)

        # HMMs don't handle "no prediction" states, so cut them out
        starts, sequences = self._split_by_unlabeled(state, 0, -1)
        for cur_start, cur_seq in zip(starts, sequences):
            predicted_states = self._model.most_likely_states(self._model_params, np.asarray([cur_seq]))
            new_states[cur_start:cur_start + len(cur_seq) + 1] = predicted_states

        # Just assign full confidence for predicted states
        # new_probs[new_states != -1] = np.eye(new_probs.shape[1])[new_states[new_states != -1]]
        new_probs = np.copy(prob)

        return new_probs, new_states

    def save(self, file: Path):
        """Saves filter settings to file.

        Args:
            file: file to write trained filter model settings
        """
        if not self._is_trained:
            raise RuntimeError('Unable to save an untrained HMM filter. Please call .train before .save.')

        payload = {
            'kwargs': self._kwargs,
            'initial_probs': self._model_params.initial.params,
            'transition_matrix': self._model_params.transitions.probs,
            'emission_probs': self._model_params.emissions.probs,
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

        self._kwargs = payload['kwargs']

        # Initialize the model
        n_classes = len(self._kwargs['class_list'])
        self._init_model(n_classes, self._kwargs['random_seed'])

        # Re-load the parameters
        params, properties = self._model.initialize(
            initial_probs=payload['initial_probs'],
            transition_matrix=payload['transition_matrix'],
            emission_probs=payload['emission_probs'],
            )
        self._model_params = params
        self._is_trained = True
