import contextlib
import getpass
import gzip
import json
import os
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import jabs.feature_extraction as fe
from jabs.pose_estimation import PoseEstimation, get_pose_path, open_pose_file
from jabs.types import ProjectDistanceUnit

from .feature_manager import FeatureManager
from .parallel_workers import FeatureLoadJobSpec, collect_labeled_features
from .prediction_manager import PredictionManager
from .project_paths import ProjectPaths
from .project_utils import to_safe_name
from .session_tracker import SessionTracker
from .settings_manager import SettingsManager
from .video_labels import VideoLabels
from .video_manager import VideoManager


def _warm_noop(n: int = 0) -> int:
    """Trivial picklable function used to force worker process spin-up."""
    return n


class Project:
    """Represents a JABS project, managing data, settings, and operations for a project directory.

    A project consists of video files, pose files, metadata, annotations, classifier data, and
    optionally predictions. This class provides methods to access and manage those resources,
    including loading/saving annotations, managing features/predictions, archiving behaviors,
    and retrieving project settings.

    Executor pool:
        Each Project owns a persistent **non-resizing** `concurrent.futures.ProcessPoolExecutor`
        used for CPU-bound feature extraction (parallelized per video). The pool size is fixed
        at construction time via ``executor_workers`` (defaults to ``os.cpu_count()`` if None).
        The pool is created lazily on first use (e.g., when calling `get_labeled_features`) or
        can be pre-spawned with `warm_executor(wait=True)` to avoid first-use latency.

        - The pool **does not auto-resize**. If you need a different size, construct a new
          `Project` with the desired ``executor_workers`` (a dedicated `resize_executor()` may
          be added later).
        - Submitting fewer jobs than workers is fine (extra workers idle). Submitting more jobs
          than workers is also fine (tasks queue).
        - Submissions are thread-safe; the pool can be used from worker/QThreads.
        - Call `shutdown_executor()` on application exit for a clean teardown. A best-effort
          shutdown is also attempted in `__del__`, but explicit shutdown is preferred.

    Args:
        project_path: Path to the project directory.
        use_cache: Whether to use cached data.
        enable_video_check: Whether to check for video file validity.
        enable_session_tracker: Whether to enable session tracking for this project.
        executor_workers: Fixed size of the process pool; if None, uses CPU count.
        validate_project_dir: Whether to validate the project directory structure on creation.

    Properties:
        dir: Project directory path.
        feature_dir: Directory for feature files.
        annotation_dir: Directory for annotation files.
        classifier_dir: Directory for classifier files.
        settings: Project metadata and preferences.
        settings_manager: SettingsManager instance for this project.
        total_project_identities: Total number of identities across all videos.
        prediction_manager: PredictionManager instance for this project.
        feature_manager: FeatureManager instance for this project.
        video_manager: VideoManager instance for this project.
        project_paths: ProjectPaths instance for this project.
    """

    def __init__(
        self,
        project_path,
        use_cache=True,
        enable_video_check=True,
        enable_session_tracker=True,
        executor_workers: int | None = None,
        validate_project_dir=True,
    ):
        self._paths = ProjectPaths(Path(project_path), use_cache=use_cache)
        self._paths.create_directories(validate=validate_project_dir)
        self._total_project_identities = 0
        self._enabled_extended_features = {}

        self._settings_manager = SettingsManager(self._paths)
        self._video_manager = VideoManager(self._paths, self._settings_manager, enable_video_check)
        self._feature_manager = FeatureManager(
            self._paths, self._video_manager.videos, self._video_manager
        )
        self._prediction_manager = PredictionManager(self)
        self._session_tracker = SessionTracker(self, tracking_enabled=enable_session_tracker)

        # write out the defaults to the project file
        if self._settings_manager.project_settings.get("defaults") != self.get_project_defaults():
            self._settings_manager.save_project_file({"defaults": self.get_project_defaults()})

        # Persistent, non-resizing process pool for feature extraction
        self._executor: ProcessPoolExecutor | None = None
        self._executor_size: int = max(1, (executor_workers or (os.cpu_count() or 1)))

        # Start a session tracker for this project.
        # Since the session has a reference to the Project, the Project should be fully initialized before starting
        # the session tracker.
        self._session_tracker.start_session()

    def _ensure_executor(self) -> ProcessPoolExecutor:
        """Create the persistent ProcessPoolExecutor once using the configured size."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self._executor_size)
        return self._executor

    def shutdown_executor(self) -> None:
        """Shut down the persistent executor (call on app exit)."""
        # We need to be defensive against partially constructed Project instances where __init__ may have
        # raised an Exception before self._executor was declared and then shutdown_executor is called by __del__.
        # In that case, the attribute may not exist, so we can't access the attribute directly here -- use
        # getattr instead.
        executor = getattr(self, "_executor", None)
        if executor is not None:
            with contextlib.suppress(Exception):
                executor.shutdown(cancel_futures=False)
            self._executor = None
            self._executor_size = 0

    def warm_executor(self, wait: bool = True) -> None:
        """Warm the project's process pool early (e.g., right after project load).

        The pool size is fixed from `__init__` and does not resize here. See the class
        docstring for details about the executor's lifecycle and guarantees.

        Args:
            wait: If True, submit trivial jobs so worker processes fully spawn before return.
        """
        executor = self._ensure_executor()
        if wait:
            futures = [executor.submit(_warm_noop, i) for i in range(self._executor_size)]
            for f in futures:
                f.result()

    def __del__(self):
        # Best-effort shutdown of persistent executor
        self.shutdown_executor()

    def _validate_pose_files(self):
        """Ensure all videos have corresponding pose files."""
        err = False
        for v in self._video_manager.videos:
            if not self.__has_pose(v):
                print(f"{v} missing pose file", file=sys.stderr)
                err = True
        if err:
            raise ValueError("Project missing pose file for one or more videos")

    @property
    def dir(self) -> Path:
        """get the project directory"""
        return self._paths.project_dir

    @property
    def feature_dir(self) -> Path:
        """get the feature directory"""
        return self._paths.feature_dir

    @property
    def annotation_dir(self) -> Path:
        """get the annotation directory"""
        return self._paths.annotations_dir

    @property
    def classifier_dir(self):
        """get the classifier directory"""
        return self._paths.classifier_dir

    @property
    def settings(self):
        """get the project metadata and preferences."""
        return self._settings_manager.project_settings

    @property
    def settings_manager(self) -> SettingsManager:
        """get the project settings manager"""
        return self._settings_manager

    @property
    def total_project_identities(self):
        """sum the number of instances across all videos in the project

        Returns:
            integer sum
        """
        return self._video_manager.total_project_identities

    @property
    def prediction_manager(self) -> PredictionManager:
        """get the prediction manager for this project"""
        return self._prediction_manager

    @property
    def feature_manager(self) -> FeatureManager:
        """get the feature manager for this project"""
        return self._feature_manager

    @property
    def video_manager(self) -> VideoManager:
        """get the video manager for this project"""
        return self._video_manager

    @property
    def project_paths(self) -> ProjectPaths:
        """get the project paths object for this project"""
        return self._paths

    def get_derived_file_paths(self, video_name: str) -> list[Path]:
        """Return a list of paths for files derived from a given video.

        Includes:
          - all files under features/<video base name>/** (recursive)
          - all files under cache/convex_hulls/<video base name>/** (recursive)
          - cache/<video base name>_pose_est_v*_cache.h5
          - predictions/<video base name>.h5
          - annotations/<video base name>.json

        Excludes:
          - video file
          - pose file

        Args:
            video_name: File name (or key) of the video in this project.

        Returns:
            List of pathlib.Path objects for all related files.
        """
        paths: list[Path] = []
        base = Path(video_name).with_suffix("").name

        # Feature files (recursive under features/<base>/)
        feature_root = self._paths.feature_dir / base
        if feature_root.exists():
            for p in feature_root.rglob("*"):
                if p.is_file():
                    paths.append(p)

        # Cached convex hulls (recursive under cache/convex_hulls/<base>/)
        if self._paths.cache_dir is not None:
            ch_root = self._paths.cache_dir / "convex_hulls" / base
            if ch_root.exists():
                for p in ch_root.rglob("*"):
                    if p.is_file():
                        paths.append(p)

            # Cached pose files: cache/<base>_pose_est_v*_cache.h5
            for p in self._paths.cache_dir.glob(f"{base}_pose_est_v*_cache.h5"):
                paths.append(p)

        # Predictions file: predictions/<base>.h5
        prediction = self._paths.prediction_dir / f"{base}.h5"
        if prediction.exists():
            paths.append(prediction)

        # Annotation file
        annotation = self._paths.annotations_dir / f"{base}.json"
        if annotation.exists():
            paths.append(annotation)

        return paths

    @property
    def labeler(self) -> str | None:
        """return name of labeler

        For now, this is just the username of the user running JABS. Return
        None if the username cannot be determined.
        """
        try:
            return getpass.getuser()
        except Exception:
            return None

    @property
    def session_tracker(self) -> SessionTracker | None:
        """get the session tracker for this project"""
        return self._session_tracker

    @staticmethod
    def is_valid_project_directory(directory: Path) -> bool:
        """Check if a directory is a valid JABS project directory.

        Currently just checks for the existence of the jabs directory and project.json file. This can be
        called before initializing a Project object, which will create these files if they do not exist which
        might not be desired behavior for tools that expect to operate on an existing JABS project directory.

        Args:
            directory: Path to the directory to check.

        Returns:
            True if the directory is a valid JABS project directory, False otherwise.
        """
        try:
            paths = ProjectPaths(directory)
            if not paths.jabs_dir.exists() or not paths.project_file.exists():
                return False
        except (ValueError, FileNotFoundError):
            return False

        return True

    def load_pose_est(self, video_path: Path) -> PoseEstimation:
        """return a PoseEstimation object for a given video path

        Args:
            video_path: pathlib.Path containing location of video file

        Returns:
            PoseEstimation object (PoseEstimationV2 or PoseEstimationV3)

        Raises:
            ValueError: if video no in project or it does not have post
                file
        """
        # ensure this video path is for a valid project video
        video_filename = Path(video_path).name
        self._video_manager.check_video_name(video_filename)

        return open_pose_file(get_pose_path(video_path), self._paths.cache_dir)

    def save_annotations(self, annotations: VideoLabels, pose: PoseEstimation):
        """save state of a VideoLabels object to the project directory

        Args:
            annotations: VideoLabels object
            pose: PoseEstimation, identity mask is used to account for dropped identity when generating intervals

        Returns:
            None
        """
        path = self._paths.annotations_dir / Path(annotations.filename).with_suffix(".json")

        annotations = annotations.as_dict(
            pose,
            project_metadata=self.settings_manager.project_metadata,
            video_metadata=self.settings_manager.video_metadata(annotations.filename),
        )
        annotations["labeler"] = self.labeler

        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(annotations, f, indent=2)
        tmp.replace(path)

        # update app version saved in project metadata if necessary
        self._settings_manager.update_version()

    def get_project_defaults(self):
        """obtain the default per-behavior settings

        Returns:
            dictionary of project settings
        """
        return self.settings_by_pose_version(
            self._feature_manager.min_pose_version,
            self._feature_manager.distance_unit,
            self._feature_manager.static_objects,
        )

    @staticmethod
    def settings_by_pose_version(
        pose_version: int = 2,
        distance_unit: ProjectDistanceUnit = ProjectDistanceUnit.PIXEL,
        static_objects: set[str] | None = None,
    ):
        """obtain project settings for a specified pose version

        Args:
            pose_version: pose version to indicate settings
            distance_unit: distance unit for settings
            static_objects: keys of static objects to include
        """
        if static_objects is None:
            static_objects = set()

        return {
            "cm_units": distance_unit,
            "window_size": fe.DEFAULT_WINDOW_SIZE,
            "social": pose_version >= 3,
            "static_objects": {
                obj: bool(pose_version >= 5 and obj in static_objects)
                for obj in fe.landmark_features.LandmarkFeatureGroup.feature_map
            },
            "segmentation": pose_version >= 6,
            "window": True,
            "fft": True,
            "balance_labels": False,
            "symmetric_behavior": False,
        }

    def save_classifier(self, classifier, behavior: str):
        """Save the classifier for the given behavior

        Args:
            classifier: the classifier to save
            behavior: string behavior name. This affects the path we save to
        """
        classifier.save(self._paths.classifier_dir / (to_safe_name(behavior) + ".pickle"))

        # update app version saved in project metadata if necessary
        self._settings_manager.update_version()

    def load_classifier(self, classifier, behavior: str):
        """Load cached classifier for the given behavior

        Args:
            classifier: the classifier to load
            behavior: string behavior name.

        Returns:
            True if load is successful and False if the file doesn't
            exist
        """
        classifier_path = self._paths.classifier_dir / (to_safe_name(behavior) + ".pickle")
        try:
            classifier.load(classifier_path)
            return True
        except OSError:
            return False

    def save_predictions(
        self,
        pose_est: PoseEstimation,
        video_name: str,
        predictions: dict[int, np.ndarray],
        probabilities: dict[int, np.ndarray],
        behavior: str,
        classifier: object,
    ) -> None:
        """Save predictions for a video in the project folder.

        Args:
            pose_est: PoseEstimation object for the video.
            video_name: name of the video these predictions correspond to.
            predictions: dict mapping identity to a 1D numpy array of predicted labels.
            probabilities: same structure as `predictions` but with floating-point values.
            behavior: string behavior name.
            classifier: Classifier object used to generate the predictions.
        """
        # set up an output filename based on the video names
        file_base = Path(video_name).with_suffix("").name + ".h5"
        output_path = self._paths.prediction_dir / file_base

        # allocate numpy arrays to write to h5 file
        prediction_labels = np.full(
            (pose_est.num_identities, pose_est.num_frames), -1, dtype=np.int8
        )
        prediction_prob = np.zeros_like(prediction_labels, dtype=np.float32)

        # stack the numpy arrays
        for identity in predictions:
            prediction_labels[identity] = predictions[identity]
            prediction_prob[identity] = probabilities[identity]

        # write to h5 file
        self._prediction_manager.write_predictions(
            behavior,
            output_path,
            prediction_labels,
            prediction_prob,
            pose_est,
            classifier,
        )

        # update app version saved in project metadata if necessary
        self._settings_manager.update_version()

    def archive_behavior(self, behavior: str):
        """Archive a behavior.

        Archives any labels for this behavior. Deletes any other files associated with this behavior.

        Args:
            behavior (str): behavior name

        Returns:
            None
        """
        safe_behavior = to_safe_name(behavior)

        # remove predictions
        path = self._paths.prediction_dir / safe_behavior
        shutil.rmtree(path, ignore_errors=True)

        # remove classifier
        path = self._paths.classifier_dir / f"{safe_behavior}.pickle"
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

        # archive labels and unfragmented_labels
        archived_labels = {}
        for video in self._video_manager.videos:
            pose = self.load_pose_est(self._video_manager.video_path(video))
            labels = self._video_manager.load_video_labels(video, pose)

            # if no labels for video skip it
            if labels is None:
                continue

            annotations = labels.as_dict(pose)

            # ensure archive structure exists for this video:
            # {
            #   "num_frames": ...,
            #   "labels": {"<behavior>": {}},
            #   "unfragmented_labels": {"<behavior>": {}}
            # }
            if video not in archived_labels:
                archived_labels[video] = {
                    "num_frames": annotations["num_frames"],
                    "labels": {},
                    "unfragmented_labels": {},
                }
            if to_safe_name(behavior) not in archived_labels[video]["labels"]:
                # keep the behavior key as provided (not safe_name) in the archive per requested schema
                archived_labels[video]["labels"][behavior] = {}
            if to_safe_name(behavior) not in archived_labels[video]["unfragmented_labels"]:
                archived_labels[video]["unfragmented_labels"][behavior] = {}

            # Move per-identity blocks for the behavior from live annotations into archive
            # Identities are stored as string keys in annotations
            # 1) Fragmented labels
            if "labels" in annotations:
                for ident in list(annotations["labels"].keys()):
                    labels = annotations["labels"].get(ident, {})
                    if behavior in labels:
                        # copy to archive
                        archived_labels[video]["labels"][behavior][ident] = labels.pop(behavior)

            # 2) Unfragmented labels
            if "unfragmented_labels" in annotations:
                for ident in list(annotations["unfragmented_labels"].keys()):
                    labels = annotations["unfragmented_labels"].get(ident, {})
                    if behavior in labels:
                        archived_labels[video]["unfragmented_labels"][behavior][ident] = (
                            labels.pop(behavior)
                        )

            # persist the modified annotations back to disk (with the behavior removed)
            self.save_annotations(VideoLabels.load(annotations, pose), pose)

        # write the archived labels out
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with gzip.open(self._paths.archive_dir / f"{safe_behavior}_{ts}.json.gz", "wt") as f:
            json.dump(archived_labels, f, indent=2)

        # save project file
        self._settings_manager.remove_behavior(behavior)

    def counts(self, behavior):
        """get the labeled frame counts and bout counts for each video in the project

        Returns:
            dict where keys are video names and values are lists of
                (
                    identity,
                    (behavior frame count - fragmented, not behavior frame count - fragmented),
                    (behavior bout count - fragmented, not behavior bout count - fragmented),
                    (behavior frame count - unfragmented, not behavior frame count - unfragmented),
                    (behavior bout count - unfragmented, not behavior bout count - unfragmented)
                )
        """
        counts = {}
        for video in self._video_manager.videos:
            counts[video] = self.load_counts(video, behavior)
        return counts

    def get_labeled_features(
        self,
        behavior: str | None = None,
        progress_callable: Callable[[], None] | None = None,
        should_terminate_callable: Callable[[], None] | None = None,
    ) -> tuple[dict, dict]:
        """Get labeled features for training (parallel per-video).

        This collects per-frame and window-based features and corresponding labels for all
        videos in the project that have annotations. Processing is parallelized at the
        **video** level using separate processes. Progress increments **once per video**.

        Note:
            This can still take a long time if features have not yet been computed.

        Args:
            behavior: Behavior name to select behavior-specific settings. If None, uses
                project defaults (all available features).
            progress_callable: If provided, it is called **once per video** when that video's
                features are collected (useful for progress bars).
            should_terminate_callable: If provided, it may be called between job submissions
                and as results complete; it should raise a `ThreadTerminatedError` if the user
                has requested early termination.

        Returns:
            tuple[dict, dict]: A tuple of (features, group_mapping).

                The first dict contains features for all labeled frames and has the keys:
                    - 'window':    pd.DataFrame of window-based features (labeled frames only)
                    - 'per_frame': pd.DataFrame of per-frame features (labeled frames only)
                    - 'labels':    np.ndarray of integer labels
                    - 'groups':    np.ndarray of group ids aligned to rows in the feature matrices

                The values in the first dict are suitable for `Classifier.leave_one_group_out()`.

                The second dict maps group ids to their source:
                    { <group id>: {'video': <video filename>, 'identity': <identity>}, ... }
        """
        # Parallel per-video feature collection using process workers.
        # Progress increments once per video.
        all_per_frame: list[pd.DataFrame] = []
        all_window: list[pd.DataFrame] = []
        all_labels: list[np.ndarray] = []
        all_group_keys: list[tuple[str, int]] = []

        # Snapshot behavior settings once
        behavior_settings = self._settings_manager.get_behavior(behavior)
        videos = list(self._video_manager.videos)

        # Early exit if no videos
        if not videos:
            return {
                "window": pd.DataFrame(),
                "per_frame": pd.DataFrame(),
                "labels": np.array([], dtype=np.int8),
                "groups": np.array([], dtype=np.int32),
            }, {}

        # Prepare per-video jobs with Path types (workers open resources)
        jobs: list[FeatureLoadJobSpec] = []
        for video in videos:
            if should_terminate_callable:
                should_terminate_callable()

            job: FeatureLoadJobSpec = {
                "video": video,
                "video_path": self._video_manager.video_path(video),
                "annotations_path": self._paths.annotations_dir / Path(video).with_suffix(".json"),
                "feature_dir": self.feature_dir,
                "cache_dir": self._paths.cache_dir,
                "behavior_settings": behavior_settings,
                "behavior_name": behavior,
            }
            jobs.append(job)

        executor = self._ensure_executor()
        # create futures and map to video names
        future_to_video = {
            executor.submit(collect_labeled_features, job): job["video"] for job in jobs
        }

        results_by_video: dict[str, dict] = {}
        for future in as_completed(future_to_video):
            # check for early exit
            if should_terminate_callable:
                should_terminate_callable()

            video_name = future_to_video[future]
            try:
                res = future.result()
            except Exception as e:
                raise RuntimeError(f"Feature collection failed for video: {video_name}") from e

            # Stage results by video for deterministic finalization
            results_by_video[video_name] = res

            if progress_callable:
                progress_callable()  # once per video

        # Deterministic finalize: append results in original 'videos' order
        for video in videos:
            if video not in results_by_video:
                continue
            res = results_by_video[video]
            all_per_frame.extend(res.get("per_frame", []))
            all_window.extend(res.get("window", []))
            all_labels.extend(res.get("labels", []))
            all_group_keys.extend(res.get("group_keys", []))

        # If nothing was produced anywhere, return empty structures
        if not (all_per_frame and all_window and all_labels):
            return {
                "window": pd.DataFrame(),
                "per_frame": pd.DataFrame(),
                "labels": np.array([], dtype=np.int8),
                "groups": np.array([], dtype=np.int32),
            }, {}

        # Build stable group ids: original video order, then identity order as observed
        key_to_gid: dict[tuple[str, int], int] = {}
        gid = 0
        for v in videos:
            seen: list[int] = []
            for video_name, ident in all_group_keys:
                if video_name == v and ident not in seen:
                    seen.append(ident)
            for ident in seen:
                key = (v, ident)
                if key not in key_to_gid:
                    key_to_gid[key] = gid
                    gid += 1

        # groups vector aligned with all_per_frame entries
        groups_list: list[np.ndarray] = [
            np.full(df.shape[0], key_to_gid[key], dtype=np.int32)
            for key, df in zip(all_group_keys, all_per_frame, strict=True)
        ]
        groups = np.concatenate(groups_list) if groups_list else np.array([], dtype=np.int32)

        group_mapping: dict[int, dict[str, int | str]] = {
            gid: {"video": v, "identity": ident} for (v, ident), gid in key_to_gid.items()
        }

        window_df = pd.concat(all_window, join="inner")
        per_frame_df = pd.concat(all_per_frame, join="inner")
        labels_arr = np.concatenate(all_labels)

        # Sanity check: ensure all outputs are aligned
        if not (len(labels_arr) == per_frame_df.shape[0] == window_df.shape[0] == groups.shape[0]):
            raise RuntimeError(
                "Mismatch among labels/per_frame/window/groups lengths: "
                f"labels={len(labels_arr)}, per_frame={per_frame_df.shape[0]}, "
                f"window={window_df.shape[0]}, groups={groups.shape[0]}"
            )

        return {
            "window": window_df,
            "per_frame": per_frame_df,
            "labels": labels_arr,
            "groups": groups,
        }, group_mapping

    def clear_cache(self):
        """clear the cache directory for this project"""
        if self._paths.cache_dir is not None:
            for f in self._paths.cache_dir.glob("*"):
                try:
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
                except OSError:
                    pass

    def __has_pose(self, vid: str):
        """check to see if a video has a corresponding pose file"""
        path = self._paths.project_dir / vid

        try:
            get_pose_path(path)
        except ValueError:
            return False
        return True

    def load_counts(self, video, behavior) -> dict[str, tuple[int, int]]:
        """load labeled frame and bout counts from json file

        Returns:
            dict of labeled frame and bout counts for each identity for
            the specified behavior.
            {
                identity: {
                    "fragmented_frame_counts": (
                        fragmented behavior frame count,
                        fragmented not behavior frame count
                    ),
                    "fragmented_bout_counts": (
                        fragmented behavior bout count,
                        fragmented not behavior bout count
                    ),
                    "unfragmented_frame_counts": (
                        unfragmented behavior frame count,
                        unfragmented not behavior frame count
                    )
                    "unfragmented_bout_counts": (
                        unfragmented behavior bout count,
                        unfragmented not behavior bout count
                    )
                }
            }

        Note: "unfragmented" counts labels where identity drops out. "fragmented" does not,
            so if an identity drops out during a bout, the bout will be split in the fragmented counts but will
            be counted as a single bout in the unfragmented counts.

        Todo:
            - with the addition of unfragmented counts, we should switch to a dict with descriptive key names instead of a tuple
        """

        def count_labels(
            behavior_labels: dict[str, list[dict]],
        ) -> tuple[tuple[int, int], tuple[int, int]]:
            blocks = behavior_labels.get(behavior, [])
            frames_behavior = 0
            frames_not_behavior = 0
            bouts_behavior = 0
            bouts_not_behavior = 0
            for b in blocks:
                if b["present"]:
                    bouts_behavior += 1
                    frames_behavior += b["end"] - b["start"] + 1
                else:
                    bouts_not_behavior += 1
                    frames_not_behavior += b["end"] - b["start"] + 1
            return (frames_behavior, frames_not_behavior), (bouts_behavior, bouts_not_behavior)

        video_filename = Path(video).name
        path = self._paths.annotations_dir / Path(video_filename).with_suffix(".json")
        counts = {}

        if path.exists():
            with path.open() as f:
                data = json.load(f)
                unfragmented_labels = data.get("unfragmented_labels", {})
                labels = data.get("labels", {})

                for identity in set(unfragmented_labels.keys()).union(labels.keys()):
                    fragmented_counts = (
                        count_labels(labels.get(identity, [])) if labels else ((0, 0), (0, 0))
                    )

                    if "unfragmented_labels" in data:
                        unfragmented_counts = count_labels(unfragmented_labels.get(identity, []))
                    else:
                        # if the file doesn't have unfragmented labels, use the fragmented counts -- they're the same
                        # unless the user creates some new labels over frames without identity
                        unfragmented_counts = fragmented_counts

                    # identity is stored as a string in the JSON file because it's used as a key. Turn it back
                    # into an int as used internally by JABS
                    counts[int(identity)] = {
                        "fragmented_frame_counts": fragmented_counts[0],
                        "fragmented_bout_counts": fragmented_counts[1],
                        "unfragmented_frame_counts": unfragmented_counts[0],
                        "unfragmented_bout_counts": unfragmented_counts[1],
                    }

        return counts

    def rename_behavior(self, old_name: str, new_name: str) -> None:
        """Rename a behavior throughout the project.

        This updates the project settings, and renames any files associated with the behavior.
        It also updates any labels for this behavior in all videos.

        Args:
            old_name (str): current behavior name
            new_name (str): new behavior name

        Returns:
            None
        """
        if new_name in self._settings_manager.behavior_names:
            raise ValueError(f"Behavior {new_name} already exists in project")

        safe_old_name = to_safe_name(old_name)
        safe_new_name = to_safe_name(new_name)

        # rename pickled classifier
        old_path = self._paths.classifier_dir / f"{safe_old_name}.pickle"
        new_path = self._paths.classifier_dir / f"{safe_new_name}.pickle"
        if old_path.exists():
            old_path.rename(new_path)

        for video in self._video_manager.videos:
            # Rename predictions dataset inside the per-video HDF5 file.
            pred_file = self._paths.prediction_dir / Path(video).with_suffix(".h5").name
            if pred_file.exists():
                with h5py.File(pred_file, "r+") as hf:
                    if "predictions" in hf and safe_old_name in hf["predictions"]:
                        grp = hf["predictions"]
                        # If dataset already exists at the destination, remove it first
                        if safe_new_name in grp:
                            del grp[safe_new_name]
                        grp.move(safe_old_name, safe_new_name)

            # rename labels inside annotation file
            if (labels := self._video_manager.load_video_labels(video)) is None:
                continue
            labels.rename_behavior(old_name, new_name)
            pose = self.load_pose_est(self._video_manager.video_path(video))
            self.save_annotations(labels, pose)

        # update project settings
        self._settings_manager.rename_behavior(old_name, new_name)
