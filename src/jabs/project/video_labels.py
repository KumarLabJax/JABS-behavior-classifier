from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from jabs.core.constants import MULTICLASS_NONE_BEHAVIOR
from jabs.pose_estimation import PoseEstimation

from .timeline_annotations import TimelineAnnotations
from .track_labels import TrackLabels

if TYPE_CHECKING:
    from .project_merge import MergeStrategy


SERIALIZED_VERSION = 1


class VideoLabels:
    """Stores and manages frame-level behavior labels for each identity in a video.

    Each identity in the video can have multiple behaviors labeled, with each (identity, behavior) pair
    corresponding to a TrackLabels object that tracks frame-wise annotations. Labels are organized such that
    each frame can be marked as NONE, BEHAVIOR, or NOT_BEHAVIOR. The class provides methods for accessing,
    counting, serializing, and loading these labels.

    Args:
        filename: Name of the video file this object represents.
        num_frames: Total number of frames in the video.

    Note:
        in several places, identities are currently handled as strings for serialization compatibility.
    """

    def __init__(self, filename: str, num_frames: int):
        self._filename = filename
        self._num_frames = num_frames
        self._identity_labels = {}
        self._annotations = TimelineAnnotations()

    @property
    def filename(self):
        """return filename of video this object represents"""
        return self._filename

    @property
    def num_frames(self):
        """return number of frames in video this object represents"""
        return self._num_frames

    @property
    def timeline_annotations(self) -> TimelineAnnotations:
        """return interval annotations for this video, if any"""
        return self._annotations

    def add_annotation(self, annotation: TimelineAnnotations.Annotation) -> None:
        """Add a non-behavior annotation to this video"""
        self._annotations.add_annotation(annotation)

    def get_track_labels(self, identity, behavior):
        """return a TrackLabels for an identity & behavior

        Args:
            identity: string representation of identity
            behavior: string behavior label

        Returns:
            TrackLabels object for this identity and behavior

        Todo:
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
            self._identity_labels[identity][behavior] = TrackLabels(self._num_frames)

        # return TrackLabels object for this identity & behavior
        return self._identity_labels[identity][behavior]

    def iter_identity_behavior_labels(self):
        """Yield ``(identity, behavior, TrackLabels)`` entries for all labeled tracks."""
        for identity, behaviors in self._identity_labels.items():
            for behavior, track_labels in behaviors.items():
                yield identity, behavior, track_labels

    def build_multiclass_label_array(self, identity: str, behavior_names: list[str]) -> np.ndarray:
        """Build a combined class-index array for multi-class label visualization.

        Maps each frame to a class index suitable for direct lookup into the
        color LUT produced by ``build_multiclass_color_lut``:

        - ``0``: unlabeled (no BEHAVIOR annotation on any track for this frame)
        - ``1``: "None" explicit-negative (BEHAVIOR on the ``MULTICLASS_NONE_BEHAVIOR`` track)
        - ``2..N+1``: behavior *i* has a BEHAVIOR annotation (index = position in
          ``behavior_names`` + 2)

        Mutual exclusivity is assumed — enforced at label-entry time — so at most
        one class will be active per frame.

        Args:
            identity: String identity index.
            behavior_names: Ordered list of project behavior names. Must not
                include the reserved ``"None"`` behavior.

        Returns:
            Integer array of shape ``(n_frames,)`` with dtype ``np.int16``.

        Raises:
            ValueError: If ``behavior_names`` contains the reserved
                ``MULTICLASS_NONE_BEHAVIOR`` name or any duplicate entries.
        """
        if MULTICLASS_NONE_BEHAVIOR in behavior_names:
            raise ValueError(
                f"behavior_names must not include the reserved name {MULTICLASS_NONE_BEHAVIOR!r}"
            )
        if len(behavior_names) != len(set(behavior_names)):
            raise ValueError("behavior_names must not contain duplicates")

        result = np.zeros(self._num_frames, dtype=np.int16)
        identity_tracks = self._identity_labels.get(identity, {})

        none_track = identity_tracks.get(MULTICLASS_NONE_BEHAVIOR)
        if none_track is not None:
            result[none_track.get_labels() == TrackLabels.Label.BEHAVIOR] = 1

        for i, behavior in enumerate(behavior_names):
            track = identity_tracks.get(behavior)
            if track is not None:
                result[track.get_labels() == TrackLabels.Label.BEHAVIOR] = i + 2

        return result

    def iter_behavior_labels(self, identity: str) -> Iterator[tuple[str, TrackLabels]]:
        """Yield ``(behavior, TrackLabels)`` for every labeled track belonging to an identity.

        Args:
            identity: String representation of the identity to iterate over.

        Yields:
            Tuples of (behavior name, TrackLabels) for each labeled track.
        """
        yield from self._identity_labels.get(identity, {}).items()

    def counts(self, behavior):
        """get the count of labeled frames and bouts for each identity in this video for a specified behavior

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

    def as_dict(
        self,
        pose: PoseEstimation,
        project_metadata: dict | None = None,
        video_metadata: dict | None = None,
    ) -> dict:
        """return dict representation of video labels

        useful for JSON serialization and saving to disk

        example return value:
        {
            "file": "filename.avi",
            "num_frames": 100,
            "external_identities: {
                "jabs identity", 1234,
            },
            "metadata": {
                "project": {},
                "video": {},
            }
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
            },
            "unfragmented_labels": {
                "jabs identity": {
                    "behavior": [
                        {
                            "start": 25,
                            "end": 50,
                            "present": True
                        }
                    ]
                }
            },
            annotations: [
                {
                    "start": 10,
                    "end": 20,
                    "tag": "annotationTag",
                    "color": "#FF0000",
                    "description": "Description for the annotation"
                },
                {
                    "start": 30,
                    "end": 40,
                    "tag": "anotherTag",
                    "color": "#00FF00",
                    "description": "Another optional description",
                    "animal_id": 0  # optional, if the annotation is associated with an identity (internal JABS ID)
                }
            ]
        }

        """
        label_dict: dict[str, Any] = {
            "version": SERIALIZED_VERSION,
            "file": self._filename,
            "num_frames": self._num_frames,
            "labels": {},
            "unfragmented_labels": {},
            "metadata": {
                "project": project_metadata if project_metadata is not None else {},
                "video": video_metadata if video_metadata is not None else {},
            },
        }

        for identity in self._identity_labels:
            label_dict["unfragmented_labels"][identity] = {}
            label_dict["labels"][identity] = {}
            for behavior in self._identity_labels[identity]:
                labels = self._identity_labels[identity][behavior]

                blocks = labels.get_blocks()
                if len(blocks):
                    label_dict["unfragmented_labels"][identity][behavior] = blocks

                blocks = labels.get_blocks(mask=pose.identity_mask(int(identity)))
                if len(blocks):
                    label_dict["labels"][identity][behavior] = blocks

        if pose.external_identities is not None:
            label_dict["external_identities"] = {}
            for i, identity in enumerate(pose.external_identities):
                # require identity to be a string for serialization
                label_dict["external_identities"][str(i)] = identity

        if len(self._annotations) > 0:
            label_dict["annotations"]: list[dict] = self._annotations.serialize()

        return label_dict

    @classmethod
    def load(cls, video_label_dict: dict, pose: PoseEstimation | None = None) -> "VideoLabels":
        """return a VideoLabels object initialized with data from a dict previously exported using the export method"""
        labels = cls(video_label_dict["file"], video_label_dict["num_frames"])

        key = "unfragmented_labels" if "unfragmented_labels" in video_label_dict else "labels"

        for identity in video_label_dict[key]:
            labels._identity_labels[identity] = {}
            for behavior in video_label_dict[key][identity]:
                labels._identity_labels[identity][behavior] = TrackLabels.load(
                    video_label_dict["num_frames"],
                    video_label_dict[key][identity][behavior],
                )

        # load non-behavior annotations if they exist
        if "annotations" in video_label_dict:
            labels._annotations = TimelineAnnotations.load(
                video_label_dict["annotations"], pose.identity_index_to_display
            )

        return labels

    def merge(self, other: "VideoLabels", strategy: "MergeStrategy") -> None:
        """Merges another VideoLabels object into this one.

        For each identity and behavior present in the other VideoLabels object, merges the corresponding
        TrackLabels into this instance according to the provided merge strategy.

        Args:
            other (VideoLabels): The VideoLabels object to merge from.
            strategy (MergeStrategy): The strategy to use when merging TrackLabels.

        Returns:
            None
        """
        for identity, behaviors in other._identity_labels.items():
            for behavior, other_track_labels in behaviors.items():
                track_labels = self.get_track_labels(identity, behavior)
                track_labels.merge(other_track_labels, strategy)

    def rename_behavior(self, old_name: str, new_name: str) -> None:
        """Rename a behavior across all identities in this VideoLabels object.

        Only renames behavior in memory; does not update any associated files on disk.
        Behavior labels must be saved to disk again after renaming to persist changes.

        Args:
            old_name (str): The current name of the behavior to rename.
            new_name (str): The new name for the behavior.

        Returns:
            None

        Raises:
            KeyError: If the old behavior name does not exist or
                if the new behavior name already exists for any identity.
        """
        # validate that new_name doesn't already exist for any identity
        for identity in self._identity_labels:
            if new_name in self._identity_labels[identity]:
                raise KeyError(f"Behavior '{new_name}' already exists for identity '{identity}'")

        for identity in self._identity_labels:
            if old_name in self._identity_labels[identity]:
                self._identity_labels[identity][new_name] = self._identity_labels[identity].pop(
                    old_name
                )
