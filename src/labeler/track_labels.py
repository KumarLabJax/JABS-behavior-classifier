import enum
from itertools import groupby
import numpy as np


class TrackLabels:
    """
    stores labels for a given identity and behavior
    """
    class Label(enum.IntEnum):
        """ label values """
        NONE = 0
        BEHAVIOR = 1
        NOT_BEHAVIOR = 2

    def __init__(self, num_frames):
        self._labels = np.zeros(num_frames, dtype=np.uint8)

    def label_behavior(self, start, end):
        """ label range [start, end] as showing behavior """
        self._labels[start:end+1] = self.Label.BEHAVIOR

    def label_not_behavior(self, start, end):
        """ label range [start, end] of frames as not showing behavior """
        self._labels[start:end+1] = self.Label.NOT_BEHAVIOR

    def clear_labels(self, start, end):
        """ clear labels for a range of frames [start, end]"""
        self._labels[start:end+1] = self.Label.NONE

    def get_frame_label(self, frame_index):
        """ get the label for a given frame """
        return self._labels[frame_index]

    def get_blocks(self):
        return self._array_to_blocks(self._labels)

    def get_slice_blocks(self, start, end):
        return self._array_to_blocks(self._labels[start:end+1])

    @classmethod
    def load(cls, num_frames, blocks):
        """
        return a TrackLabels object initialized with data from a list of blocks
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
            :return:  list of blocks of frames that have been labeled as having
            the behavior or not having the behavior. Each block has the following
            representation:
            {
                'start': block_start_frame,
                'end': block_end_frame,
                'present': boolean
            }
            where 'present' is True if the block has been labeled as showing the
            behavior and False if it has been labeled as not showing the behavior.
            Unlabeled frames are not included, so the total number of frames is
            also required to reconstruct the labels array.
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