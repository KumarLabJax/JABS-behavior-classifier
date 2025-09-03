import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from intervaltree import Interval, IntervalTree

MAX_TAG_LEN = 32


class TimelineAnnotations:
    """A stand-alone helper class for managing timeline annotations stored in an IntervalTree.

    This class provides methods to load, serialize, query, and manipulate annotations
    represented as intervals within an IntervalTree. Annotations typically represent
    labeled segments on a timeline, each with start and end frames, tags, rendering color,
    and optional identity indices.

    Attributes:
        _tree (IntervalTree): The underlying interval tree storing annotation intervals.
    """

    @dataclass
    class Annotation:
        """Dataclass to define an Annotation to be stored in the IntervalTree."""

        start: int
        end: int
        tag: str
        color: str
        description: str | None = None
        identity_index: int | None = None
        display_identity: str | None = None

    def __init__(self) -> None:
        """Initialize a TimelineAnnotations instance with an empty IntervalTree."""
        self._tree: IntervalTree = IntervalTree()

    def add_annotation(self, annotation: Annotation) -> None:
        """Add a new timeline annotation to the interval tree.

        Args:
            annotation (Annotation): The annotation to add.
        """
        self._tree[annotation.start : annotation.end + 1] = {
            "tag": annotation.tag,
            "color": annotation.color,
            "description": annotation.description,
            "identity": annotation.identity_index,
            "display_identity": annotation.display_identity,
        }

    @classmethod
    def load(
        cls, data: list[dict[str, Any]], id_index_to_display: Callable[[int], str] | None = None
    ) -> "TimelineAnnotations":
        """Load a TimelineAnnotations instance from a JSON-compatible dictionary.

        Args:
            data (dict): The dictionary containing serialized timeline annotations.
            id_index_to_display (Callable[[int], str] | None): Optional function to map identity index to a display string.

        Returns:
            TimelineAnnotations: An instance loaded with the provided data.

        Note: loading currently skips invalid entries with a warning printed to stderr. Consider
        raising an exception for stricter handling in the future.
        """
        annotations = cls()

        for annotation in data:
            try:
                start = annotation["start"]
                end = annotation["end"]
                tag = annotation["tag"]
                color = annotation["color"]
            except KeyError:
                print(
                    "Missing required annotation fields, loading skipped for annotation:",
                    annotation,
                    file=sys.stderr,
                )
                continue

            # validate the tag format:
            if len(tag) < 1 or len(tag) > MAX_TAG_LEN:
                print(
                    f"Annotation tag must be 1 to {MAX_TAG_LEN} characters in length, skipping annotation: \n\t{annotation}",
                    file=sys.stderr,
                )
                continue
            # only allow alphanumeric characters, underscores, and hyphens
            if not all(c.isalnum() or c in "_-" for c in tag):
                print(
                    f"Annotation tag can only contain alphanumeric characters, underscores, and hyphens. Skipping annotation: \n\t{annotation}",
                    file=sys.stderr,
                )
                continue

            # Create a data dict for the interval.
            # Note: description and identity are optional fields
            identity_index = annotation.get("identity")
            if identity_index is not None:
                if id_index_to_display:
                    display_identity = id_index_to_display(identity_index)
                else:
                    display_identity = str(identity_index)
            else:
                display_identity = None
            data = {
                "tag": tag,
                "color": color,
                "description": annotation.get("description"),
                "identity": identity_index,
                "display_identity": display_identity,
            }

            # Add the annotation to the IntervalTree.
            # The start and end contained in the JSON file are inclusive, so we add 1 to end.
            annotations._tree[start : end + 1] = data
        return annotations

    def serialize(self) -> list[dict]:
        """Convert the internal IntervalTree to a JSON-serializable list of dictionaries.

        Returns:
            list[dict]: A list containing a dictionary representation for each timeline annotation,
             suitable for JSON serialization.
        """
        annotations = []
        for element in self._tree:
            try:
                annotation_data = {
                    "start": element.begin,
                    "end": element.end - 1,  # convert to inclusive
                    "tag": element.data["tag"],
                    "color": element.data["color"],
                }
            except KeyError as e:
                print(f"Missing required annotation data: {e}")
                continue

            # optional fields
            description = element.data.get("description")
            if description is not None:
                annotation_data["description"] = description
            identity = element.data.get("identity")
            if identity is not None:
                annotation_data["identity"] = identity

            annotations.append(annotation_data)
        return annotations

    @staticmethod
    def _interval_matches(
        interval: Interval, *, start: int, end: int, tag: str, identity_index: int | None
    ) -> bool:
        """Determine if an interval matches the given annotation parameters.

        Args:
            interval (Interval): The interval to check.
            start (int): The expected start frame.
            end (int): The expected end frame (inclusive).
            tag (str): The annotation tag to match (case-insensitive).
            identity_index (int | None): The identity index to match, or None.

        Returns:
            bool: True if the interval matches all parameters, False otherwise.
        """
        if interval.begin != start or (interval.end - 1) != end:
            return False
        data = interval.data or {}
        return data["tag"].lower() == tag.lower() and data.get("identity") == identity_index

    def find_matching_intervals(
        self, *, start: int, end: int, tag: str, identity_index: int | None
    ) -> list[Interval]:
        """Find all intervals matching the specified annotation parameters.

        Args:
            start (int): The start frame of the annotation.
            end (int): The end frame of the annotation (inclusive).
            tag (str): The annotation tag to match (case-insensitive).
            identity_index (int | None): The identity index to match, or None.

        Returns:
            list[Interval]: A list of matching Interval objects.
        """
        candidates = self._tree[start : end + 1]
        return [
            interval
            for interval in candidates
            if self._interval_matches(
                interval, start=start, end=end, tag=tag, identity_index=identity_index
            )
        ]

    def annotation_exists(
        self, *, start: int, end: int, tag: str, identity_index: int | None
    ) -> bool:
        """Check if an annotation with the given parameters exists in the interval tree.

        Args:
            start (int): The start frame of the annotation.
            end (int): The end frame of the annotation (inclusive).
            tag (str): The annotation tag to match (case-insensitive).
            identity_index (int | None): The identity index to match, or None.

        Returns:
            bool: True if such an annotation exists, False otherwise.
        """
        return bool(
            self.find_matching_intervals(
                start=start, end=end, tag=tag, identity_index=identity_index
            )
        )

    def remove_annotation_by_key(
        self, *, start: int, end: int, tag: str, identity_index: int | None
    ) -> int:
        """Remove all annotations matching the specified parameters from the interval tree.

        Args:
            start (int): The start frame of the annotation.
            end (int): The end frame of the annotation (inclusive).
            tag (str): The annotation tag to match (case-insensitive).
            identity_index (int | None): The identity index to match, or None.

        Returns:
            int: The number of annotations removed.

        Note: this implementation removes all matching annotations -- however in practice
        it should match 0 or 1 since we check for key collision at insertion time.
        """
        intervals = self.find_matching_intervals(
            start=start, end=end, tag=tag, identity_index=identity_index
        )
        for interval in intervals:
            self._tree.remove(interval)
        return len(intervals)

    def __len__(self) -> int:
        """Return the number of annotations stored in the interval tree.

        Returns:
            int: The number of annotations in the tree.
        """
        return len(self._tree)

    def __getitem__(self, key):
        """Provide array-like access to annotations.

        Supports slicing and indexing by delegating to the internal IntervalTree's __getitem__.

        Args:
            key (int, slice, or interval specifier): The key to index or slice the annotations.

        Returns:
            set[Interval]: The intervals corresponding to the key.
        """
        return self._tree[key]
