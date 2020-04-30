from .track_labels import TrackLabels


class VideoLabels:
    """
    store the labels associated with a video file.

    labels are organized by "identity". Each identity may have multiple
    behaviors labeled. An identity and behavior uniquely identifies a
    TrackLabels object, which stores labels for each frame in the video. Each
    frame can have one of three label values: TrackLabels.Label.NONE,
    Tracklabels.Label.BEHAVIOR, and TrackLabels.Label.NOT_BEHAVIOR
    """
    def __init__(self, filename, num_frames):
        self._filename = filename
        self._num_frames = num_frames
        self._identity_labels = {}

    @property
    def filename(self):
        return self._filename

    def get_track_labels(self, identity, behavior):
        """ return a TrackLabels for an identity & behavior """
        identity_labels = self._identity_labels.get(identity)

        if identity_labels is None:
            self._identity_labels[identity] = {}

        track_labels = self._identity_labels[identity].get(behavior)

        if track_labels is None:
            self._identity_labels[identity][behavior] = \
                TrackLabels(self._num_frames)

        return self._identity_labels[identity][behavior]

    def as_dict(self):
        """
        return dict representation of self

        example return value:
        {
            "file": "filename.avi",
            "num_frames": 100,
            "labels": {
                "identity name": {
                    "behavior name": [
                        {
                            "start": 25,
                            "end": 50,
                            "present": True
                        }
                    ]
                }
            }
        }

        """
        labels = {}
        for identity in self._identity_labels:
            labels[identity] = {}
            for behavior in self._identity_labels[identity]:
                labels[identity][behavior] = \
                    self._identity_labels[identity][behavior].get_blocks()

        return {
            'file': self._filename,
            'num_frames': self._num_frames,
            'labels': labels
        }

    @classmethod
    def load(cls, video_label_dict):
        """
        return a VideoLabels object initialized with data from a dict previously
        exported using the export() method
        """
        labels = cls(video_label_dict['file'], video_label_dict['num_frames'])
        for identity in video_label_dict['labels']:
            labels._identity_labels[identity] = {}
            for behavior in video_label_dict['labels'][identity]:
                labels._identity_labels[identity][behavior] = TrackLabels.load(
                    video_label_dict['num_frames'],
                    video_label_dict['labels'][identity][behavior])

        return labels
