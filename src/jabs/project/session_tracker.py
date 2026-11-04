import datetime
import enum
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .project import Project


class ActivityTypes(enum.IntEnum):
    """Enum for different types of activity tracked in a session.

    Activity types:
    - BEHAVIOR_LABEL_CREATED: user created a behavior label for an interval of frames
    - NOT_BEHAVIOR_LABEL_CREATED: user created a not-behavior label for an interval of frames
    - BEHAVIOR_SELECTED: user selected a behavior from the behavior list
    - LABEL_DELETED: user deleted a label for an interval of frames
    - VIDEO_OPENED: user opened a video file
    - VIDEO_CLOSED: user closed a video file
    - CLASSIFIER_TRAINED: user trained a classifier
    - SESSION_PAUSED: labeling session was paused (e.g. user minimized the application)
    - SESSION_RESUMED: labeling session was resumed
    """

    BEHAVIOR_LABEL_CREATED = enum.auto()
    NOT_BEHAVIOR_LABEL_CREATED = enum.auto()
    BEHAVIOR_SELECTED = enum.auto()
    LABEL_DELETED = enum.auto()
    VIDEO_OPENED = enum.auto()
    VIDEO_CLOSED = enum.auto()
    CLASSIFIER_TRAINED = enum.auto()
    SESSION_PAUSED = enum.auto()
    SESSION_RESUMED = enum.auto()


class SessionTracker:
    """Tracks activity for the current labeling session.

    Args:
        project (Project): The project instance to track activity for.
        tracking_enabled (bool): Whether to enable tracking of activities. Defaults to True.
          If False, all session tracking methods will be no-ops.

    Activity tracked includes:
    - session start time: timestamp the project was opened
    - session end time: timestamp of when the project was closed (by opening a different project or closing the application)
    - labels created: each time a user applies a label to a range of frames
    - labels deleted: each time a user removes a label from a range of frames
    - video opened: each time a user opens a video file
    - video closed: each time a user closes a video file
    - classifier trained: each time a user trains a classifier
    """

    def __init__(self, project: "Project", tracking_enabled: bool = True):
        self._project = project
        self._session_file: Path | None = None
        self._session: dict | None = None
        self._tracking_enabled = tracking_enabled

    def __del__(self):
        """Ensure session log is completed and flushed when the tracker is deleted

        This will finalize the session log when the user opens a new Project or closes the application.
        """
        if self._tracking_enabled and self._session_file and self._session:
            self.end_session()

    @property
    def enabled(self) -> bool:
        """Returns whether session tracking is enabled."""
        return self._tracking_enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Sets whether session tracking is enabled."""
        # If disabling session tracking, end the session if it exists
        if not value:
            self.end_session()

        self._tracking_enabled = value
        if value and self._session is None:
            # session tracking was just enabled for the first time, initialize the tracking session
            self.start_session()

    def start_session(self) -> None:
        """Starts a new labeling session.

        This will create a new file in the project/jabs/session/ directory. This is separate from __init__() because
        it's up to the Project class to start the session tracking after it's been fully initialized.
        """
        if not self._tracking_enabled:
            return

        timestamp = datetime.datetime.now(datetime.timezone.utc)

        self._session = {
            "session_start": timestamp.isoformat(),
            "session_end": None,
            "starting_label_counts": {},
            "ending_label_counts": {},
            "activity_log": [],
        }

        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        self._session_file = self._project.project_paths.session_dir / f"{timestamp_str}.json"
        self._flush_session()

    def end_session(self) -> None:
        """Ends the current labeling session.

        Logs session end time and flushes the session log to the session file.
        """
        if not self._tracking_enabled or not self._session:
            return

        self._session["session_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # get the ending counts for each behavior
        for behavior in self._session["starting_label_counts"]:
            self._session["ending_label_counts"][behavior] = self._project_label_counts(behavior)

        self._flush_session()
        self._session = None
        self._session_file = None

    def behavior_selected(self, behavior_name: str) -> None:
        """Log the selection of a behavior.

        The first time a behavior is selected during a session its project-wide
        label counts are also snapshotted, so the session can report how much was
        labeled while it was open.

        Args:
            behavior_name: The name of the behavior selected.
        """
        if not self._tracking_enabled or not self._session:
            return

        self._append_activity(ActivityTypes.BEHAVIOR_SELECTED, {"behavior": behavior_name})

        if behavior_name not in self._session["starting_label_counts"]:
            self._session["starting_label_counts"][behavior_name] = self._project_label_counts(
                behavior_name
            )

        self._flush_session()

    def label_created(
        self,
        video_path: Path,
        identity: int,
        behavior_name: str,
        present: bool,
        start_frame: int,
        end_frame: int,
    ) -> None:
        """Log the creation of a label.

        Args:
            video_path: The video file where the label was created.
            identity: The identity of the individual being labeled.
            behavior_name: The name of the behavior being labeled.
            present: Whether the behavior is present in the specified frame range
              (behavior label = True, not behavior label = False).
            start_frame: The starting frame of the label.
            end_frame: The ending frame of the label (inclusive).
        """
        self._log_activity(
            ActivityTypes.BEHAVIOR_LABEL_CREATED
            if present
            else ActivityTypes.NOT_BEHAVIOR_LABEL_CREATED,
            {
                "video": video_path.name,
                "behavior": behavior_name,
                "identity": identity,
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
        )

    def label_deleted(
        self, video_path: Path, identity: int, behavior_name: str, start_frame: int, end_frame: int
    ) -> None:
        """Log the deletion of a label.

        Args:
            video_path: The video file where the label was deleted.
            identity: The identity of the individual being labeled.
            behavior_name: The name of the behavior being labeled.
            start_frame: The starting frame of the label.
            end_frame: The ending frame of the label (inclusive).
        """
        self._log_activity(
            ActivityTypes.LABEL_DELETED,
            {
                "video": video_path.name,
                "identity": identity,
                "behavior": behavior_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
            },
        )

    def video_opened(self, video_path: Path) -> None:
        """Log the opening of a video file.

        Args:
            video_path: The video file that was opened.
        """
        self._log_activity(ActivityTypes.VIDEO_OPENED, {"video": video_path.name})

    def video_closed(self, video_path: Path) -> None:
        """Log the closing of a video file.

        Args:
            video_path: The video file that was closed.
        """
        self._log_activity(ActivityTypes.VIDEO_CLOSED, {"video": video_path.name})

    def classifier_trained(
        self,
        behavior_name: str,
        classifier_type: str,
        k: int,
        accuracy: float | None = None,
        fbeta_behavior: float | None = None,
    ) -> None:
        """Log the training of a classifier.

        Args:
            behavior_name: The behavior the classifier was trained for.
            classifier_type: Name of the classifier type that was trained.
            k: Number of cross-validation folds used.
            accuracy: Optional mean cross-validation accuracy. Left out of the log
                entry when not given.
            fbeta_behavior: Optional mean f-beta score for the behavior class.
                Left out of the log entry when not given.
        """
        fields: dict[str, Any] = {
            "behavior": behavior_name,
            "classifier_type": classifier_type,
            "k-fold": k,
        }

        if accuracy is not None:
            fields["mean accuracy"] = f"{accuracy:.3}"
        if fbeta_behavior is not None:
            fields["mean fbeta (behavior)"] = f"{fbeta_behavior:.3}"

        self._log_activity(ActivityTypes.CLASSIFIER_TRAINED, fields)

    def pause_session(self) -> None:
        """Log the pausing of a labeling session."""
        self._log_activity(ActivityTypes.SESSION_PAUSED)

    def resume_session(self) -> None:
        """Log the resuming of a labeling session."""
        self._log_activity(ActivityTypes.SESSION_RESUMED)

    def _append_activity(
        self, activity_type: ActivityTypes, fields: dict[str, Any] | None = None
    ) -> None:
        """Append one timestamped entry to the in-memory activity log.

        Every entry shares the same envelope - a UTC timestamp, the activity
        name, then the activity's own fields - so it is built here rather than
        spelled out once per activity, which is how the entries would drift apart
        in key naming or timestamp format.

        The entry is not written to disk; the caller flushes once it has finished
        updating the session. Callers must check ``_tracking_enabled`` and
        ``_session`` themselves - use :meth:`_log_activity` unless there is more
        session state to update before flushing.

        Args:
            activity_type: Activity being recorded; its ``name`` is stored.
            fields: Activity-specific keys, added after ``activity`` in the order
                given.
        """
        self._session["activity_log"].append(
            {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "activity": activity_type.name,
                **(fields or {}),
            }
        )

    def _log_activity(
        self, activity_type: ActivityTypes, fields: dict[str, Any] | None = None
    ) -> None:
        """Record one activity and flush the session, when tracking is active.

        This is the entire body of most of the public activity methods: nothing
        is recorded when tracking is disabled or no session is open, and an entry
        that is recorded is flushed immediately so the log survives an unclean
        exit.

        Args:
            activity_type: Activity being recorded; its ``name`` is stored.
            fields: Activity-specific keys, added after ``activity`` in the order
                given.
        """
        if not self._tracking_enabled or not self._session:
            return

        self._append_activity(activity_type, fields)
        self._flush_session()

    def _project_label_counts(self, behavior: str) -> dict[str, int]:
        """Total the project's unfragmented label counts for one behavior.

        The session log records these totals twice: once when a behavior is first
        selected (``starting_label_counts``) and again when the session ends
        (``ending_label_counts``). Both come from here so the two snapshots are
        always totaled the same way and stay comparable.

        Args:
            behavior: Name of the behavior to total counts for.

        Returns:
            Mapping with ``frames_behavior``, ``frames_not_behavior``,
            ``bouts_behavior`` and ``bouts_not_behavior`` totals, summed over
            every video and identity in the project.
        """
        frames_behavior = 0
        frames_not_behavior = 0
        bouts_behavior = 0
        bouts_not_behavior = 0

        for video_counts in self._project.counts(behavior).values():
            for identity_count in video_counts.values():
                frames_behavior += identity_count["unfragmented_frame_counts"][0]
                frames_not_behavior += identity_count["unfragmented_frame_counts"][1]
                bouts_behavior += identity_count["unfragmented_bout_counts"][0]
                bouts_not_behavior += identity_count["unfragmented_bout_counts"][1]

        return {
            "frames_behavior": frames_behavior,
            "frames_not_behavior": frames_not_behavior,
            "bouts_behavior": bouts_behavior,
            "bouts_not_behavior": bouts_not_behavior,
        }

    def _flush_session(self):
        """Flush the current session data to the session file."""
        if self._session_file and self._session:
            try:
                tmp = self._session_file.with_suffix(".json.tmp")
                with tmp.open("w") as f:
                    json.dump(self._session, f, indent=4)
                tmp.replace(self._session_file)
            except NameError:
                # Python is shutting down, and open is no longer available
                pass
