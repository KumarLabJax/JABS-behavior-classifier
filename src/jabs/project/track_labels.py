import enum
import math
from itertools import groupby

import numpy as np


class TrackLabels:
    """
    Stores and manages frame-level labels for a single identity and behavior.

    Each frame can be labeled as NONE, BEHAVIOR, or NOT_BEHAVIOR. The class provides methods to set, clear,
    and query labels over frame ranges, as well as utilities for counting labeled frames and bouts, exporting
    label blocks, and downsampling label arrays for visualization or analysis.

    Args:
        num_frames (int): Number of frames to allocate for label storage.
    """

    class Label(enum.IntEnum):
        """label values"""

        NONE = -1
        NOT_BEHAVIOR = 0
        BEHAVIOR = 1

        # the following only used in down sampling label array
        # they have special meaning
        MIX = 2
        PAD = 3

    def __init__(self, num_frames):
        self._labels = np.full(num_frames, self.Label.NONE.value, dtype=np.byte)

    def label_behavior(self, start, end, mask=None):
        """label range [start, end] as showing behavior"""
        self._set_labels(start, end, self.Label.BEHAVIOR, mask)

    def label_not_behavior(self, start, end, mask=None):
        """label range [start, end] of frames as not showing behavior"""
        self._set_labels(start, end, self.Label.NOT_BEHAVIOR, mask)

    def clear_labels(self, start, end):
        """clear labels for a range of frames [start, end]"""
        self._labels[start : end + 1] = self.Label.NONE

    def _set_labels(self, start, end, label, mask=None):
        """set label value for a range of frames

        Args:
            start: start of range, inclusive
            end: end of range, inclusive
            label: label to apply to frames
            mask: optional mask array, if present only set values where
                the mask array is not zero

        Returns:
            None
        """
        if mask is not None:
            self._labels[start : end + 1][mask != 0] = label
        else:
            self._labels[start : end + 1] = label

    def get_labels(self):
        """Return the full label array for all frames.

        Todo:
         - make this a property
        """
        return self._labels

    def get_frame_label(self, frame_index):
        """get the label for a given frame"""
        return self._labels[frame_index]

    @property
    def label_count(self):
        """Return a tuple with the count of frames for each label class.

        Returns:
            tuple: (number of frames labeled as BEHAVIOR, number of frames labeled as NOT_BEHAVIOR)
        """
        return (
            np.count_nonzero(self._labels == self.Label.BEHAVIOR),
            np.count_nonzero(self._labels == self.Label.NOT_BEHAVIOR),
        )

    @property
    def bout_count(self):
        """Return a tuple with the number of contiguous bouts for each label class.

        Returns:
            tuple: (number of behavior bouts, number of not behavior bouts)
                - A bout is a contiguous block of frames labeled as BEHAVIOR or NOT_BEHAVIOR.
        """
        blocks = self._array_to_blocks(self._labels)
        bouts_behavior = 0
        bouts_not_behavior = 0

        for b in blocks:
            if b["present"]:
                bouts_behavior += 1
            else:
                bouts_not_behavior += 1
        return bouts_behavior, bouts_not_behavior

    @property
    def counts(self):
        """return the label and bout counts"""
        return self.label_count, self.bout_count

    def get_blocks(self):
        """get blocks for entire label array

        see _array_to_blocks() for return type
        """
        return self._array_to_blocks(self._labels)

    def get_slice_blocks(self, start, end):
        """get label blocks for a slice of frames

        block start and end frame numbers will be relative to the slice start
        """
        return self._array_to_blocks(self._labels[start : end + 1])

    @classmethod
    def downsample(cls, labels: np.ndarray, size: int):
        """Downsample a label array to a specified size using custom binning rules.

        Each output element summarizes a bin of input frames:
            - Label.NONE: All frames in the bin are NONE.
            - Label.BEHAVIOR: All frames are BEHAVIOR or NONE.
            - Label.NOT_BEHAVIOR: All frames are NOT_BEHAVIOR or NONE.
            - Label.MIX: Bin contains both BEHAVIOR and NOT_BEHAVIOR.
            - Label.PAD: Bin consists entirely of padding added to fit the output size.

        Args:
            labels (np.ndarray): Input label array.
            size (int): Desired output array size.

        Returns:
            np.ndarray: Downsampled label array of length `size`.
        """

        def bincount(array):
            return {
                cls.Label.NONE: np.count_nonzero(array == cls.Label.NONE.value),
                cls.Label.BEHAVIOR: np.count_nonzero(array == cls.Label.BEHAVIOR.value),
                cls.Label.NOT_BEHAVIOR: np.count_nonzero(
                    array == cls.Label.NOT_BEHAVIOR.value
                ),
            }

        # we may need to pad the label array if it is not evenly divisible by
        # the new size
        pad_size = math.ceil(labels.size / size) * size - labels.size

        # create the padded array
        padded = np.append(labels, np.full(pad_size, cls.Label.PAD.value))

        # split the padded array into 'size' bins each with 'bin_size' values
        bin_size = padded.size // size
        binned = padded.reshape(-1, bin_size)

        # create output array
        downsampled = np.empty(size, dtype=np.byte)

        # fill output array
        for i in range(size):
            counts = bincount(binned[i])

            if counts[cls.Label.NONE] == len(binned[i]):
                downsampled[i] = cls.Label.NONE.value
            elif (
                counts[cls.Label.BEHAVIOR] != 0 and counts[cls.Label.NOT_BEHAVIOR] == 0
            ):
                downsampled[i] = cls.Label.BEHAVIOR.value
            elif (
                counts[cls.Label.NOT_BEHAVIOR] != 0 and counts[cls.Label.BEHAVIOR] == 0
            ):
                downsampled[i] = cls.Label.NOT_BEHAVIOR.value
            elif (
                counts[cls.Label.NOT_BEHAVIOR] != 0 and counts[cls.Label.BEHAVIOR] != 0
            ):
                downsampled[i] = cls.Label.MIX.value
            else:
                downsampled[i] = cls.Label.PAD.value

        return downsampled

    @classmethod
    def load(cls, num_frames: int, blocks: list[dict]) -> "TrackLabels":
        """Create a TrackLabels object from a list of labeled blocks.

        Args:
            num_frames (int): Total number of frames in the video.
            blocks (list[dict]): List of label blocks, where each block is a dict with
                'start', 'end', and 'present' fields as produced by get_blocks().

        Returns:
            TrackLabels: Initialized TrackLabels object with labels set according to the provided blocks.
        """
        labels = cls(num_frames)
        for block in blocks:
            if block["present"]:
                labels.label_behavior(block["start"], block["end"])
            else:
                labels.label_not_behavior(block["start"], block["end"])
        return labels

    @classmethod
    def _array_to_blocks(cls, array: np.ndarray) -> list[dict]:
        """Convert a label array into a list of labeled blocks for export or serialization.

        Args:
            array (np.ndarray): Numpy array of TrackLabels.Label values for each frame.

        Returns:
            list[dict]: Each dict represents a contiguous block of labeled frames (excluding NONE), with:
                - 'start': Start frame index of the block.
                - 'end': End frame index of the block.
                - 'present': True if the block is labeled as BEHAVIOR, False if labeled as NOT_BEHAVIOR.

            Unlabeled (NONE) frames are omitted from the output.
        """
        block_start = 0
        blocks = []

        for val, group in groupby(array):
            count = len([*group])
            if val != cls.Label.NONE:
                blocks.append(
                    {
                        "start": block_start,
                        "end": block_start + count - 1,
                        # note: val == cls.Label.BEHAVIOR returns a numpy bool, which is not json serializable
                        "present": bool(val == cls.Label.BEHAVIOR),
                    }
                )
            block_start += count
        return blocks
