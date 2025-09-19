import datetime
import enum
import json
from pathlib import Path
from typing import TYPE_CHECKING

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
            counts = self._project.counts(behavior)

            frames_behavior = 0
            frames_not_behavior = 0
            bouts_behavior = 0
            bouts_not_behavior = 0

            for video_counts in counts.values():
                for identity_count in video_counts.values():
                    frames_behavior += identity_count["unfragmented_frame_counts"][0]
                    frames_not_behavior += identity_count["unfragmented_frame_counts"][1]
                    bouts_behavior += identity_count["unfragmented_bout_counts"][0]
                    bouts_not_behavior += identity_count["unfragmented_bout_counts"][1]

            self._session["ending_label_counts"][behavior] = {
                "frames_behavior": frames_behavior,
                "frames_not_behavior": frames_not_behavior,
                "bouts_behavior": bouts_behavior,
                "bouts_not_behavior": bouts_not_behavior,
            }

        self._flush_session()
        self._session = None
        self._session_file = None

    def behavior_selected(self, behavior_name: str) -> None:
        """Log the selection of a behavior.

        Args:
            behavior_name (str): The name of the behavior selected.
        """
        if not self._tracking_enabled or not self._session:
            return

        activity = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activity": ActivityTypes.BEHAVIOR_SELECTED.name,
            "behavior": behavior_name,
        }
        self._session["activity_log"].append(activity)

        if behavior_name not in self._session["starting_label_counts"]:
            counts = self._project.counts(behavior_name)

            frames_behavior = 0
            frames_not_behavior = 0
            bouts_behavior = 0
            bouts_not_behavior = 0

            for video in counts:
                for _, identity_count in counts[video].items():
                    frames_behavior += identity_count["unfragmented_frame_counts"][0]
                    frames_not_behavior += identity_count["unfragmented_frame_counts"][1]
                    bouts_behavior += identity_count["unfragmented_bout_counts"][0]
                    bouts_not_behavior += identity_count["unfragmented_bout_counts"][1]

            self._session["starting_label_counts"][behavior_name] = {
                "frames_behavior": frames_behavior,
                "frames_not_behavior": frames_not_behavior,
                "bouts_behavior": bouts_behavior,
                "bouts_not_behavior": bouts_not_behavior,
            }

        self._flush_session()

    def label_created(
        self,
        video_path: Path,
        identity: int,
        behavior_name: str,
        present: bool,
        start_frame: int,
        end_frame: int,
    ):
        """Log the creation of a label.

        Args:
            video_path (str): The video file where the label was created.
            identity (int): The identity of the individual being labeled.
            behavior_name (str): The name of the behavior being labeled.
            present (bool): Whether the behavior is present in the specified frame range
              (behavior label = True, not behavior label = False).
            start_frame (int): The starting frame of the label.
            end_frame (int): The ending frame of the label (inclusive).
        """
        if not self._tracking_enabled or not self._session:
            return

        activity = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activity": ActivityTypes.BEHAVIOR_LABEL_CREATED.name
            if present
            else ActivityTypes.NOT_BEHAVIOR_LABEL_CREATED.name,
            "video": video_path.name,
            "behavior": behavior_name,
            "identity": identity,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        self._session["activity_log"].append(activity)
        self._flush_session()

    def label_deleted(
        self, video_path: Path, identity: int, behavior_name: str, start_frame: int, end_frame: int
    ):
        """Log the deletion of a label.

        Args:
            video_path (Path): The video file where the label was deleted.
            identity (int): The identity of the individual being labeled.
            behavior_name (str): The name of the behavior being labeled.
            start_frame (int): The starting frame of the label.
            end_frame (int): The ending frame of the label (inclusive).
        """
        if not self._tracking_enabled or not self._session:
            return

        activity = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activity": ActivityTypes.LABEL_DELETED.name,
            "video": video_path.name,
            "identity": identity,
            "behavior": behavior_name,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        self._session["activity_log"].append(activity)
        self._flush_session()

    def video_opened(self, video_path: Path):
        """Log the opening of a video file."""
        if not self._tracking_enabled or not self._session:
            return

        activity = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activity": ActivityTypes.VIDEO_OPENED.name,
            "video": video_path.name,
        }
        self._session["activity_log"].append(activity)
        self._flush_session()

    def video_closed(self, video_path: Path):
        """Log the closing of a video file."""
        if not self._tracking_enabled or not self._session:
            return

        activity = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activity": ActivityTypes.VIDEO_CLOSED.name,
            "video_path": video_path.name,
        }
        self._session["activity_log"].append(activity)
        self._flush_session()

    def classifier_trained(
        self,
        behavior_name: str,
        classifier_type: str,
        k: int,
        accuracy: float | None = None,
        fbeta_behavior: float | None = None,
        fbeta_notbehavior: float | None = None,
    ):
        """Log the training of a classifier."""
        if not self._tracking_enabled or not self._session:
            return

        activity = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activity": ActivityTypes.CLASSIFIER_TRAINED.name,
            "behavior": behavior_name,
            "classifier_type": classifier_type,
            "k-fold": k,
        }

        if accuracy is not None:
            activity["mean accuracy"] = f"{accuracy:.3}"
        if fbeta_behavior is not None:
            activity["mean fbeta (behavior)"] = f"{fbeta_behavior:.3}"
        if fbeta_notbehavior is not None:
            activity["mean fbeta (not behavior)"] = f"{fbeta_notbehavior:.3}"

        self._session["activity_log"].append(activity)
        self._flush_session()

    def pause_session(self):
        """Log the pausing of a labeling session."""
        if not self._tracking_enabled or not self._session:
            return

        activity = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activity": ActivityTypes.SESSION_PAUSED.name,
        }
        self._session["activity_log"].append(activity)
        self._flush_session()

    def resume_session(self):
        """Log the resuming of a labeling session."""
        if not self._tracking_enabled or not self._session:
            return

        activity = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "activity": ActivityTypes.SESSION_RESUMED.name,
        }
        self._session["activity_log"].append(activity)
        self._flush_session()

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
