from .track_labels import TrackLabels


class VideoLabels:
    """store the labels associated with a video file.

    labels are organized by "identity". Each identity may have multiple
    behaviors labeled. An identity and behavior uniquely identifies a
    TrackLabels object, which stores labels for each frame in the video. Each
    frame can have one of three label values: TrackLabels.Label.NONE,
    Tracklabels.Label.BEHAVIOR, and TrackLabels.Label.NOT_BEHAVIOR

    TODO stop using str for identities in method parameters, switch to int
    """
    def __init__(self, filename, num_frames, external_identities: list[int] | None = None):
        self._filename = filename
        self._num_frames = num_frames
        self._identity_labels = {}
        self._external_identities = external_identities

    @property
    def filename(self):
        """return filename of video this object represents"""
        return self._filename

    @property
    def num_frames(self):
        return self._num_frames

    def get_track_labels(self, identity, behavior):
        """return a TrackLabels for an identity & behavior

        Args:
            identity: string representation of identity
            behavior: string behavior label

        Returns:
            TrackLabels object for this identity and behavior

        TODO:
            handle integer identity
        """

        # require identity to be a string for serialization
        if not isinstance(identity, str):
            raise ValueError("Identity must be a string")

        identity_labels = self._identity_labels.get(identity)

        # identity not already present
        if identity_labels is None:
            self._identity_labels[identity] = {}

        track_labels = self._identity_labels[identity].get(behavior)

        # identity doesn't have annotations for this behavior, create a new
        # TrackLabels object
        if track_labels is None:
            self._identity_labels[identity][behavior] = \
                TrackLabels(self._num_frames)

        # return TrackLabels object for this identity & behavior
        return self._identity_labels[identity][behavior]

    def counts(self, behavior):
        """get the count of labeled frames and bouts for each identity in this
        video for a specified behavior

        Args:
            behavior: behavior to get label counts for

        Returns:
            list of tuples with the following form
            (
                identity,
                (behavior frame count, not behavior frame count),
                (behavior bout count, not behavior bout count)
            )
        """
        counts = []
        for identity in self._identity_labels:
            if behavior in self._identity_labels[identity]:
                c = self._identity_labels[identity][behavior].counts
                counts.append((identity, c[0], c[1]))
        return counts

    def as_dict(self) -> dict:
        """return dict representation of self, useful for JSON serialization and
        saving to disk or caching in memory without storing the full
        numpy label array when user switches to a different video

        example return value:
        {
            "file": "filename.avi",
            "num_frames": 100,
            "external_identities: {
                "jabs identity", 1234,
            },
            "labels": {
                "jabs identity": {
                    "behavior": [
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

        label_dict = {
            'file': self._filename,
            'num_frames': self._num_frames,
            "labels": {},
        }

        for identity in self._identity_labels:
            label_dict["labels"][identity] = {}
            for behavior in self._identity_labels[identity]:
                blocks = self._identity_labels[identity][behavior].get_blocks()
                if len(blocks):
                    label_dict["labels"][identity][behavior] = blocks


        if self._external_identities is not None:
            label_dict['external_identities'] = {}
            for i, identity in enumerate(self._external_identities):
                label_dict['external_identities'][str(i)] = identity

        return label_dict

    @classmethod
    def load(cls, video_label_dict):
        """return a VideoLabels object initialized with data from a dict previously
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
