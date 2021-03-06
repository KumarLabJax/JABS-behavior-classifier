import enum
import math
from itertools import groupby

import numpy as np


class TrackLabels:
    """
    Stores labels for a given identity and behavior. Requires one byte per
    frame to store labels (e.g. approx. 108KB per 1 hour of 30fps video)
    """

    class Label(enum.IntEnum):
        """ label values """
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
        """ label range [start, end] as showing behavior """
        self._set_labels(start, end, self.Label.BEHAVIOR, mask)

    def label_not_behavior(self, start, end, mask=None):
        """ label range [start, end] of frames as not showing behavior """
        self._set_labels(start, end, self.Label.NOT_BEHAVIOR, mask)

    def clear_labels(self, start, end):
        """ clear labels for a range of frames [start, end] """
        self._labels[start:end+1] = self.Label.NONE

    def _set_labels(self, start, end, label, mask=None):
        """
        set label value for a range of frames
        :param start: start of range, inclusive
        :param end: end of range, inclusive
        :param label: label to apply to frames
        :param mask: optional mask array, if present only set values where
        the mask array is not zero
        :return: None
        """
        if mask is not None:
            self._labels[start:end + 1][mask != 0] = label
        else:
            self._labels[start:end + 1] = label

    def get_labels(self):
        return self._labels

    def get_frame_label(self, frame_index):
        """ get the label for a given frame """
        return self._labels[frame_index]

    @property
    def label_count(self):
        """
        property that returns a tuple with the count of the number of frames
        for each label class
        :return: (count of frames labeled as showing behavior,
                  count of frames labeled as not showing behavior)
        """
        return (np.count_nonzero(self._labels == self.Label.BEHAVIOR),
                np.count_nonzero(self._labels == self.Label.NOT_BEHAVIOR))

    @property
    def bout_count(self):
        """
        property that returns a tuple with the count of the number of bouts
        of each label class
        :return: (count of bouts of behavior,
                  count of bouts of "not behavior")
        """
        blocks = self._array_to_blocks(self._labels)
        bouts_behavior = 0
        bouts_not_behavior = 0

        for b in blocks:
            if b['present']:
                bouts_behavior += 1
            else:
                bouts_not_behavior += 1
        return bouts_behavior, bouts_not_behavior

    @property
    def counts(self):
        """
        return the label and bout counts
        """
        return self.label_count, self.bout_count

    def get_blocks(self):
        """
        get blocks for entire label array
        see _array_to_blocks() for return type
        """
        return self._array_to_blocks(self._labels)

    def get_slice_blocks(self, start, end):
        """
        get label blocks for a slice of frames
        block start and end frame numbers will be relative to the slice start
        """
        return self._array_to_blocks(self._labels[start:end+1])

    @classmethod
    def downsample(cls, labels, size):
        """
        Downsample the label array for a "zoomed out" view. We use a custom
        downsampling algorithm. Each element in the downsampled array is
        assigned one of the following values:
            Label.NONE: all elements in the bin have the value of Label.NONE
            Label.BEHAVIOR: all elements are either zero or Label.BEHAVIOR
            Label.NOT_BEHAVIOR: all elements are either zero or Label.NOT_BEHAVIOR
            Label.MIX: bin contains Label.BEHAVIOR and Label.NOT_BEHAVIOR
            Label.PAD: bin consists entirely of padding added to input array to
            make it evenly divisible by output size
        :param labels: numpy label array
        :param size: size of the resulting downsampled label array
        :return: numpy array of size 'size' with downsampled values
        """

        def bincount(array):
            return {
                cls.Label.NONE: np.count_nonzero(array == cls.Label.NONE.value),
                cls.Label.BEHAVIOR: np.count_nonzero(array == cls.Label.BEHAVIOR.value),
                cls.Label.NOT_BEHAVIOR: np.count_nonzero(array == cls.Label.NOT_BEHAVIOR.value)
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
            elif counts[cls.Label.BEHAVIOR] != 0 and counts[cls.Label.NOT_BEHAVIOR] == 0:
                downsampled[i] = cls.Label.BEHAVIOR.value
            elif counts[cls.Label.NOT_BEHAVIOR] != 0 and counts[cls.Label.BEHAVIOR] == 0:
                downsampled[i] = cls.Label.NOT_BEHAVIOR.value
            elif counts[cls.Label.NOT_BEHAVIOR] != 0 and counts[cls.Label.BEHAVIOR] != 0:
                downsampled[i] = cls.Label.MIX.value
            else:
                downsampled[i] = cls.Label.PAD.value

        return downsampled

    @classmethod
    def load(cls, num_frames, blocks):
        """
        return a TrackLabels object initialized with data from a list of blocks
        :param num_frames total number of frames in the video
        :param blocks - blocks to use to initialize frame label array. see
        _array_to_blocks() for format
        :return initialized TrackLabels object
        """
        labels = cls(num_frames)
        for block in blocks:
            if block['present']:
                labels.label_behavior(block['start'], block['end'])
            else:
                labels.label_not_behavior(block['start'], block['end'])
        return labels

    @classmethod
    def _array_to_blocks(cls, array):
        """
            return label blocks as something that can easily be exported as json
            for saving to disk
            :param array numpy label array to encode as blocks. Each element
            should be one of TrackLabels.Label enum values.
            :return:  list of blocks of frames that have been labeled as having
            the behavior or not having the behavior. Each block has the
            following representation:
            {
                'start': block_start_frame,
                'end': block_end_frame,
                'present': boolean
            }
            where 'present' is True if the block has been labeled as showing the
            behavior and False if it has been labeled as not showing the
            behavior. Unlabeled frames are not included, so the total number of
            frames is also required to reconstruct the labels array.
        """

        block_start = 0
        blocks = []

        for val, group in groupby(array):
            count = len([*group])
            if val != cls.Label.NONE:
                blocks.append({
                    'start': block_start,
                    'end': block_start + count - 1,
                    'present': True if val == cls.Label.BEHAVIOR else False
                })
            block_start += count
        return blocks
