import contextlib
import getpass
import gzip
import json
import logging
import shutil
import sys
from collections.abc import Callable
from concurrent.futures import as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

import jabs.feature_extraction as fe
from jabs.core.constants import CACHE_FORMAT_KEY, MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import (
    CacheFormat,
    ClassifierMode,
    CrossValidationGroupingStrategy,
    ProjectDistanceUnit,
)
from jabs.pose_estimation import (
    PoseEstimation,
    get_pose_file_major_version,
    get_pose_path,
    open_pose_file,
)

from .feature_manager import FeatureManager
from .parallel_workers import (
    BinaryFeatureLoadJobSpec,
    MulticlassFeatureLoadJobSpec,
    VideoScanJobSpec,
    VideoScanResult,
    collect_binary_labeled_features,
    collect_multiclass_labeled_features,
    scan_video_metadata,
)
from .prediction_manager import MULTICLASS_PREDICTION_KEY, PredictionManager
from .project_paths import ProjectPaths
from .project_utils import to_safe_name
from .session_tracker import SessionTracker
from .settings_manager import SettingsManager
from .track_labels import TrackLabels
from .video_labels import VideoLabels
from .video_manager import VideoManager

logger = logging.getLogger(__name__)

MULTICLASS_CLASSIFIER_FILENAME = "_multiclass.pickle"

if TYPE_CHECKING:
    from jabs.classifier import ClassifierProtocol
    from jabs.core.utils.process_pool_manager import ProcessPoolManager


class Project:
    """Represents a JABS project, managing data, settings, and operations for a project directory.

    A project consists of video files, pose files, metadata, annotations, classifier data, and
    optionally predictions. This class provides methods to access and manage those resources,
    including loading/saving annotations, managing features/predictions, archiving behaviors,
    and retrieving project settings.

    Executor pool:
        The Project can optionally use a shared application-level `ProcessPoolManager` for CPU-bound
        feature extraction (parallelized per video). If not provided (None), operations run single-threaded.

        - When pool is provided, it's managed at the application level
        - Submissions are thread-safe; the pool can be used from worker/QThreads
        - The GUI passes a shared pool; CLI scripts typically run single-threaded (pool=None)

    Args:
        project_path: Path to the project directory.
        process_pool: Optional shared ProcessPoolManager for feature extraction. If None, runs single-threaded.
        use_cache: Whether to use cached data.
        enable_video_check: Whether to check for video file validity.
        enable_session_tracker: Whether to enable session tracking for this project.
        validate_project_dir: Whether to validate the project directory structure on creation.
        video_dir: Optional directory containing video files. Defaults to `project_path`.
        pose_dir: Optional directory containing pose files. Defaults to `project_path`.

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
        process_pool: "ProcessPoolManager | None" = None,
        use_cache=True,
        enable_video_check=True,
        enable_session_tracker=True,
        validate_project_dir=True,
        video_dir: Path | None = None,
        pose_dir: Path | None = None,
    ):
        self._paths = ProjectPaths(
            Path(project_path),
            use_cache=use_cache,
            video_dir=Path(video_dir) if video_dir is not None else None,
            pose_dir=Path(pose_dir) if pose_dir is not None else None,
        )
        self._paths.create_directories(validate=validate_project_dir)
        self._total_project_identities = 0
        self._enabled_extended_features = {}

        # Capture whether project.json exists before SettingsManager loads it.
        # Used below to choose the cache_format default: Parquet for brand-new projects,
        # HDF5 for existing projects that predate the setting (conservative back-compat).
        is_new_project = not self._paths.project_file.exists()

        self._settings_manager = SettingsManager(self._paths)
        scan_results = self._run_video_scan(enable_video_check, process_pool)
        self._video_manager = VideoManager(
            self._paths, self._settings_manager, enable_video_check, scan_results=scan_results
        )
        self._feature_manager = FeatureManager(
            self._paths,
            self._video_manager.videos,
            self._video_manager,
            scan_results=scan_results,
        )
        self._prediction_manager = PredictionManager(self)
        self._session_tracker = SessionTracker(self, tracking_enabled=enable_session_tracker)

        # write out the defaults to the project file
        if self._settings_manager.project_settings.get("defaults") != self.get_project_defaults():
            self._settings_manager.save_project_file({"defaults": self.get_project_defaults()})

        # Persist cache_format. New projects default to Parquet; existing projects that
        # predate this setting default to HDF5 to preserve backward compatibility.
        existing_settings = dict(self._settings_manager.project_settings.get("settings", {}))
        if CACHE_FORMAT_KEY not in existing_settings:
            default_format = CacheFormat.PARQUET if is_new_project else CacheFormat.HDF5
            existing_settings[CACHE_FORMAT_KEY] = default_format.value
            self._settings_manager.save_project_file({"settings": existing_settings})

        # Shared application-level process pool for feature extraction
        self._process_pool = process_pool

        # Start a session tracker for this project.
        # Since the session has a reference to the Project, the Project should be fully initialized before starting
        # the session tracker.
        self._session_tracker.start_session()

    def _run_video_scan(
        self,
        enable_video_check: bool,
        process_pool: "ProcessPoolManager | None",
    ) -> dict[str, VideoScanResult]:
        """Collect per-video metadata via a parallel scan of all pose HDF5 files.

        Discovers videos and their pose files, then dispatches one
        :func:`~jabs.project.parallel_workers.scan_video_metadata` worker per
        video. Workers open each HDF5 file exactly once and return everything
        :class:`VideoManager` and :class:`FeatureManager` need at load time.

        When ``process_pool`` is ``None`` (e.g. CLI scripts), workers run
        sequentially — still reduces HDF5 opens compared to the legacy path.

        Videos whose pose file cannot be located are skipped here;
        :class:`VideoManager` will raise the appropriate error during its own
        ``_validate_pose_files`` check.

        Args:
            enable_video_check: When ``True``, workers also read video frame
                counts (mirrors the ``enable_video_check`` flag).
            process_pool: Optional shared process pool for parallel execution.

        Returns:
            Mapping from video filename to scan result.
        """
        videos = sorted(VideoManager.get_videos(self._paths.video_dir))
        jobs: list[VideoScanJobSpec] = []
        for video in videos:
            video_path = self._paths.video_dir / video
            try:
                pose_path = get_pose_path(video_path, self._paths.pose_dir)
            except ValueError:
                # Missing pose file — VideoManager._validate_pose_files will report it.
                continue
            major_version = get_pose_file_major_version(pose_path)
            jobs.append(
                VideoScanJobSpec(
                    video=video,
                    video_path=video_path,
                    pose_path=pose_path,
                    pose_major_version=major_version,
                    scan_frame_counts=enable_video_check,
                )
            )

        if not jobs:
            return {}

        if process_pool is not None:
            future_to_video = {
                process_pool.submit(scan_video_metadata, job): job["video"] for job in jobs
            }
            results: dict[str, VideoScanResult] = {}
            for future in as_completed(future_to_video):
                result: VideoScanResult = future.result()
                results[result["video"]] = result
            return results

        return {job["video"]: scan_video_metadata(job) for job in jobs}

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

    @property
    def cache_format(self) -> CacheFormat:
        """Get the feature cache format for this project.

        Returns:
            The configured ``CacheFormat``. The setting is always written to
            ``project.json`` on first open, so this property should never read an
            absent value in practice. Falls back to ``CacheFormat.HDF5`` for
            unrecognized values.
        """
        raw = self._settings_manager.project_settings.get("settings", {}).get(
            CACHE_FORMAT_KEY, CacheFormat.HDF5.value
        )
        try:
            return CacheFormat(raw)
        except ValueError:
            logger.error(
                "Unrecognized cache_format value %r in project.json; falling back to HDF5", raw
            )
            return CacheFormat.HDF5

    def clear_feature_cache(self) -> None:
        """Delete all feature cache files for every identity in the project.

        Removes HDF5 and Parquet cache files from each per-identity directory
        under the features directory. The directories themselves are preserved.
        The new format will be written on the next cache miss.
        """
        from jabs.io.feature_cache import clear_cache

        feature_dir = self._paths.feature_dir
        if not feature_dir.exists():
            return
        for video_dir in feature_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for sub in video_dir.iterdir():
                if not sub.is_dir():
                    continue
                try:
                    int(sub.name)
                    # sub is an identity directory (flat layout: features/<video>/<id>)
                    clear_cache(sub)
                except ValueError:
                    # sub is a pose-hash directory (hash layout: features/<video>/<hash>/<id>)
                    for identity_dir in sub.iterdir():
                        if identity_dir.is_dir():
                            clear_cache(identity_dir)

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

        return open_pose_file(
            self._video_manager.get_cached_pose_path(video_filename),
            self._paths.cache_dir,
        )

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

    def _classifier_path(self, behavior: str | None = None) -> Path:
        """Return the classifier path for the current classifier mode.

        Args:
            behavior: Behavior name for binary classifiers. Ignored in multi-class mode.

        Returns:
            Path to the classifier file.

        Raises:
            ValueError: If binary mode is active and no behavior name is provided.
        """
        if self.settings_manager.classifier_mode == ClassifierMode.MULTICLASS:
            return self._paths.classifier_dir / MULTICLASS_CLASSIFIER_FILENAME

        if behavior is None:
            raise ValueError("behavior is required when saving or loading binary classifiers")

        return self._paths.classifier_dir / f"{to_safe_name(behavior)}.pickle"

    def save_classifier(
        self,
        classifier: "ClassifierProtocol",
        behavior: str | None = None,
    ) -> None:
        """Save the classifier for the current classifier mode.

        Args:
            classifier: Classifier to save.
            behavior: Behavior name for binary classifiers. Ignored in multi-class mode.

        Raises:
            ValueError: If binary mode is active and no behavior name is provided.
        """
        classifier.save(self._classifier_path(behavior))

        # update app version saved in project metadata if necessary
        self._settings_manager.update_version()

    def load_classifier(
        self,
        classifier: "ClassifierProtocol",
        behavior: str | None = None,
    ) -> bool:
        """Load cached classifier for the current classifier mode.

        Args:
            classifier: Classifier to load into.
            behavior: Behavior name for binary classifiers. Ignored in multi-class mode.

        Returns:
            True if load is successful and False if the file does not exist.

        Raises:
            ValueError: If binary mode is active and no behavior name is provided.
        """
        classifier_path = self._classifier_path(behavior)
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
        classifier: "ClassifierProtocol",
        postprocessed_predictions: dict[int, np.ndarray] | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        """Save predictions for a video in the project folder.

        Args:
            pose_est: PoseEstimation object for the video.
            video_name: name of the video these predictions correspond to.
            predictions: dict mapping identity to a 1D numpy array of predicted labels.
            probabilities: same structure as `predictions` but with floating-point values.
            behavior: string behavior name.
            classifier: Classifier object used to generate the predictions.
            postprocessed_predictions: dict mapping identity to a 1D numpy array of predicted labels after
                post-processing has been applied. If provided, these will be saved alongside the
                raw predictions.
            class_names: Optional ordered class names for multiclass predictions.
        """
        # set up an output filename based on the video names
        file_base = Path(video_name).with_suffix("").name + ".h5"
        output_path = self._paths.prediction_dir / file_base

        # allocate numpy arrays to write to h5 file
        prediction_labels = np.full(
            (pose_est.num_identities, pose_est.num_frames), -1, dtype=np.int8
        )
        # Probability shape is determined by mode, not by sniffing a sample
        # array: multi-class predictions (class_names provided) store one column
        # per class, binary predictions store a scalar per frame. Deciding from
        # class_names keeps the shape correct even when `probabilities` is empty
        # (e.g. a video with no identities to classify).
        prediction_prob: np.ndarray
        if class_names is not None:
            prediction_prob = np.zeros(
                (pose_est.num_identities, pose_est.num_frames, len(class_names)),
                dtype=np.float32,
            )
        else:
            prediction_prob = np.zeros_like(prediction_labels, dtype=np.float32)

        # Expected per-identity probability shape; checked explicitly before
        # assignment because allocating from class_names (rather than from the
        # array itself) means a mis-shaped input could otherwise broadcast
        # silently (e.g. (n_frames, 1) duplicated across classes).
        expected_prob_shape = prediction_prob.shape[1:]

        if postprocessed_predictions:
            postprocessed_labels = np.full(
                (pose_est.num_identities, pose_est.num_frames), -1, dtype=np.int8
            )
        else:
            postprocessed_labels = None

        # stack the numpy arrays
        for identity in predictions:
            identity_prob = probabilities[identity]
            if identity_prob.shape != expected_prob_shape:
                raise ValueError(
                    f"probability array for identity {identity} has shape "
                    f"{identity_prob.shape}, expected {expected_prob_shape}"
                )
            prediction_labels[identity] = predictions[identity]
            prediction_prob[identity] = identity_prob
            if postprocessed_predictions:
                postprocessed_labels[identity] = postprocessed_predictions[identity]

        # write to h5 file
        self._prediction_manager.write_predictions(
            behavior,
            output_path,
            prediction_labels,
            prediction_prob,
            pose_est,
            classifier,
            postprocessed_predictions=postprocessed_labels,
            class_names=class_names,
        )

        # update app version saved in project metadata if necessary
        self._settings_manager.update_version()

    def get_overlapping_behavior_label_videos(self) -> list[str]:
        """Return filenames of videos containing frames labeled with multiple behaviors.

        Scans every video in the project for annotation conflicts where a single
        identity has the same frame labeled BEHAVIOR for two or more behaviors
        simultaneously. Includes the reserved "None" behavior track, consistent
        with how classifier_utils.merge_labels() detects conflicts at training time.

        Returns:
            Sorted list of video filenames containing at least one overlap.
            An empty list means no conflicts exist.
        """
        conflicting: list[str] = []
        for video in self._video_manager.videos:
            if (labels := self._video_manager.load_video_labels(video)) is None:
                continue
            identities = {identity for identity, _, _ in labels.iter_identity_behavior_labels()}
            video_has_conflict = False
            for identity in identities:
                behavior_counts: npt.NDArray[np.intp] | None = None
                for _, track in labels.iter_behavior_labels(identity):
                    behavior_mask = (track.get_labels() == TrackLabels.Label.BEHAVIOR).astype(
                        np.intp
                    )
                    if behavior_counts is None:
                        behavior_counts = behavior_mask
                    else:
                        behavior_counts = behavior_counts + behavior_mask
                        if np.any(behavior_counts > 1):
                            video_has_conflict = True
                            break
                if video_has_conflict:
                    conflicting.append(video)
                    break
        return sorted(conflicting)

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
            if (labels := self._video_manager.load_video_labels(video)) is None:
                continue

            pose = self.load_pose_est(self._video_manager.video_path(video))
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

    def _build_feature_load_job_base(self, video: str, behavior_settings: dict) -> dict:
        """Construct the per-video fields shared by every feature-load job spec."""
        return {
            "video": video,
            "video_path": self._video_manager.video_path(video),
            "pose_path": self._video_manager.get_cached_pose_path(video),
            "annotations_path": self._paths.annotations_dir / Path(video).with_suffix(".json"),
            "feature_dir": self.feature_dir,
            "cache_dir": self._paths.cache_dir,
            "behavior_settings": behavior_settings,
            "cache_format": self.cache_format.value,
        }

    def _collect_features_parallel(
        self,
        jobs: list[dict],
        worker_fn: Callable[[dict], dict],
        progress_callable: Callable[[], None] | None,
        should_terminate_callable: Callable[[], None] | None,
    ) -> dict[str, dict]:
        """Run feature-collection jobs in parallel (or single-threaded fallback).

        Args:
            jobs: One job spec per video.
            worker_fn: Worker function to apply to each job spec.
            progress_callable: Called once per completed video, if provided.
            should_terminate_callable: Called between submissions and completions,
                if provided; should raise on user-requested cancellation.

        Returns:
            Dict keyed by video name. Callers reorder by their canonical
            ``videos`` list for deterministic concatenation.
        """
        executor = self._process_pool
        results_by_video: dict[str, dict] = {}

        if executor is not None:
            future_to_video = {executor.submit(worker_fn, job): job["video"] for job in jobs}
            for future in as_completed(future_to_video):
                if should_terminate_callable:
                    should_terminate_callable()
                video_name = future_to_video[future]
                try:
                    res = future.result()
                except Exception as e:
                    raise RuntimeError(f"Feature collection failed for video: {video_name}") from e
                results_by_video[video_name] = res
                if progress_callable:
                    progress_callable()
        else:
            for job in jobs:
                if should_terminate_callable:
                    should_terminate_callable()
                try:
                    res = worker_fn(job)
                except Exception as e:
                    raise RuntimeError(
                        f"Feature collection failed for video: {job['video']}"
                    ) from e
                results_by_video[job["video"]] = res
                if progress_callable:
                    progress_callable()

        return results_by_video

    @staticmethod
    def _assign_cv_group_ids(
        all_group_keys: list[tuple[str, int]],
        videos: list[str],
        grouping_strategy: CrossValidationGroupingStrategy,
    ) -> tuple[dict[tuple[str, int], int], dict[int, dict]]:
        """Assign deterministic cross-validation group ids.

        Args:
            all_group_keys: ``(video, identity)`` tuples in row order.
            videos: Canonical list of project videos; ids are assigned in this order.
            grouping_strategy: ``INDIVIDUAL`` groups one (video, identity) pair per
                gid; ``VIDEO`` groups all identities of a video together.

        Returns:
            Tuple of ``(key_to_gid, group_mapping)`` where ``key_to_gid`` maps each
            ``(video, identity)`` pair to its group id and ``group_mapping`` maps
            each group id back to ``{"video": ..., "identity": ...}``.
        """
        key_to_gid: dict[tuple[str, int], int] = {}
        group_mapping: dict[int, dict] = {}
        gid = 0
        if grouping_strategy == CrossValidationGroupingStrategy.INDIVIDUAL:
            for v in videos:
                seen: list[int] = []
                for video_name, ident in all_group_keys:
                    if video_name == v and ident not in seen:
                        seen.append(ident)
                for ident in seen:
                    key = (v, ident)
                    if key not in key_to_gid:
                        key_to_gid[key] = gid
                        group_mapping[gid] = {"video": v, "identity": ident}
                        gid += 1
        elif grouping_strategy == CrossValidationGroupingStrategy.VIDEO:
            video_to_gid: dict[str, int] = {}
            for v in videos:
                if v not in video_to_gid:
                    video_to_gid[v] = gid
                    group_mapping[gid] = {"video": v, "identity": None}
                    gid += 1
                for video_name, ident in all_group_keys:
                    if video_name == v:
                        key_to_gid[(v, ident)] = video_to_gid[v]
        else:
            raise ValueError(f"Unknown grouping strategy: {grouping_strategy}")
        return key_to_gid, group_mapping

    @staticmethod
    def _build_groups_array(
        all_group_keys: list[tuple[str, int]],
        all_per_frame: list[pd.DataFrame],
        key_to_gid: dict[tuple[str, int], int],
    ) -> np.ndarray:
        """Build the per-row group id array aligned to concatenated feature matrices."""
        groups_list: list[np.ndarray] = [
            np.full(df.shape[0], key_to_gid[key], dtype=np.int32)
            for key, df in zip(all_group_keys, all_per_frame, strict=True)
        ]
        return np.concatenate(groups_list) if groups_list else np.array([], dtype=np.int32)

    def get_labeled_features(
        self,
        behavior: str | None = None,
        progress_callable: Callable[[], None] | None = None,
        should_terminate_callable: Callable[[], None] | None = None,
        grouping_strategy: CrossValidationGroupingStrategy | None = None,
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
            grouping_strategy: Optional override for cross-validation grouping strategy. If None, uses project settings.

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
        behavior_settings = self._settings_manager.get_behavior(behavior)
        videos = list(self._video_manager.videos)
        if grouping_strategy is None:
            grouping_strategy = self.settings_manager.cv_grouping_strategy

        if not videos:
            return {
                "window": pd.DataFrame(),
                "per_frame": pd.DataFrame(),
                "labels": np.array([], dtype=np.int8),
                "groups": np.array([], dtype=np.int32),
            }, {}

        jobs: list[BinaryFeatureLoadJobSpec] = []
        for video in videos:
            if should_terminate_callable:
                should_terminate_callable()
            job: BinaryFeatureLoadJobSpec = {
                **self._build_feature_load_job_base(video, behavior_settings),
                "behavior_name": behavior,
            }
            jobs.append(job)

        results_by_video = self._collect_features_parallel(
            jobs,
            collect_binary_labeled_features,
            progress_callable,
            should_terminate_callable,
        )

        all_per_frame: list[pd.DataFrame] = []
        all_window: list[pd.DataFrame] = []
        all_labels: list[np.ndarray] = []
        all_group_keys: list[tuple[str, int]] = []
        for video in videos:
            if video not in results_by_video:
                continue
            res = results_by_video[video]
            all_per_frame.extend(res["per_frame"])
            all_window.extend(res["window"])
            all_labels.extend(res["labels"])
            all_group_keys.extend(res["group_keys"])

        if not (all_per_frame and all_window and all_labels):
            return {
                "window": pd.DataFrame(),
                "per_frame": pd.DataFrame(),
                "labels": np.array([], dtype=np.int8),
                "groups": np.array([], dtype=np.int32),
            }, {}

        key_to_gid, group_mapping = self._assign_cv_group_ids(
            all_group_keys, videos, grouping_strategy
        )
        groups = self._build_groups_array(all_group_keys, all_per_frame, key_to_gid)
        window_df = pd.concat(all_window, join="inner")
        per_frame_df = pd.concat(all_per_frame, join="inner")
        labels_arr = np.concatenate(all_labels)

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

    def get_multiclass_labeled_features(
        self,
        progress_callable: Callable[[], None] | None = None,
        should_terminate_callable: Callable[[], None] | None = None,
        grouping_strategy: CrossValidationGroupingStrategy | None = None,
        behavior_settings: dict[str, object] | None = None,
    ) -> tuple[dict, dict]:
        """Get multiclass-labeled features for training (parallel per-video).

        In multiclass mode, frames are included only when they have an explicit
        ``TrackLabels.Label.BEHAVIOR`` label in at least one class track (including
        ``MULTICLASS_NONE_BEHAVIOR``).

        Args:
            progress_callable: Called once per completed video; used to drive progress bars.
            should_terminate_callable: If provided, called between jobs; should raise
                ``ThreadTerminatedError`` on user cancellation.
            grouping_strategy: Optional override for cross-validation grouping strategy.
                If None, uses project settings.
            behavior_settings: Feature-extraction settings (must include ``window_size``).
                If None, falls back to ``get_project_defaults()``.

        Returns:
            tuple[dict, dict]: A tuple of ``(features, group_mapping)``.
                ``features`` has keys:
                - ``window``: pd.DataFrame of window-based features
                - ``per_frame``: pd.DataFrame of per-frame features
                - ``labels_by_behavior``: dict[str, np.ndarray] of aligned labels
                - ``groups``: np.ndarray of group ids
        """
        behavior_names = list(self.settings_manager.behavior_names)
        if behavior_settings is None:
            behavior_settings = self.get_project_defaults()
        videos = list(self._video_manager.videos)
        if grouping_strategy is None:
            grouping_strategy = self.settings_manager.cv_grouping_strategy

        if not videos:
            return {
                "window": pd.DataFrame(),
                "per_frame": pd.DataFrame(),
                "labels_by_behavior": {},
                "groups": np.array([], dtype=np.int32),
            }, {}

        jobs: list[MulticlassFeatureLoadJobSpec] = []
        for video in videos:
            if should_terminate_callable:
                should_terminate_callable()
            job: MulticlassFeatureLoadJobSpec = {
                **self._build_feature_load_job_base(video, behavior_settings),
                "behavior_names": behavior_names,
            }
            jobs.append(job)

        results_by_video = self._collect_features_parallel(
            jobs,
            collect_multiclass_labeled_features,
            progress_callable,
            should_terminate_callable,
        )

        all_per_frame: list[pd.DataFrame] = []
        all_window: list[pd.DataFrame] = []
        all_labels_by_behavior: dict[str, list[np.ndarray]] = {}
        all_group_keys: list[tuple[str, int]] = []
        for video in videos:
            if video not in results_by_video:
                continue
            res = results_by_video[video]
            per_frame_items = res["per_frame"]
            window_items = res["window"]
            labels_by_behavior_items = res["labels_by_behavior"]
            group_keys_items = res["group_keys"]

            if not (
                len(per_frame_items)
                == len(window_items)
                == len(labels_by_behavior_items)
                == len(group_keys_items)
            ):
                raise RuntimeError(
                    "Mismatch in multiclass worker result lengths: "
                    f"per_frame={len(per_frame_items)}, "
                    f"window={len(window_items)}, "
                    f"labels_by_behavior={len(labels_by_behavior_items)}, "
                    f"group_keys={len(group_keys_items)}"
                )

            all_per_frame.extend(per_frame_items)
            all_window.extend(window_items)
            all_group_keys.extend(group_keys_items)
            # Append per-identity label slices in the same order as per_frame/window
            # entries so they stay row-aligned after concatenation.
            for labels_by_behavior in labels_by_behavior_items:
                for name, arr in labels_by_behavior.items():
                    all_labels_by_behavior.setdefault(name, []).append(arr)

        if not (all_per_frame and all_window and all_group_keys):
            return {
                "window": pd.DataFrame(),
                "per_frame": pd.DataFrame(),
                "labels_by_behavior": {},
                "groups": np.array([], dtype=np.int32),
            }, {}

        key_to_gid, group_mapping = self._assign_cv_group_ids(
            all_group_keys, videos, grouping_strategy
        )
        groups = self._build_groups_array(all_group_keys, all_per_frame, key_to_gid)
        window_df = pd.concat(all_window, join="inner")
        per_frame_df = pd.concat(all_per_frame, join="inner")
        n_rows = per_frame_df.shape[0]
        labels_by_behavior_arr = {
            name: np.concatenate(arrays) if arrays else np.array([], dtype=np.int8)
            for name, arrays in all_labels_by_behavior.items()
        }

        # Ensure every expected behavior key is present so downstream consumers
        # can rely on a stable key set even if a behavior had no labels anywhere.
        expected_labels = {MULTICLASS_NONE_BEHAVIOR, *behavior_names}
        for missing in expected_labels.difference(labels_by_behavior_arr.keys()):
            labels_by_behavior_arr[missing] = np.full(
                n_rows, TrackLabels.Label.NONE, dtype=np.int8
            )

        if not (n_rows == window_df.shape[0] == groups.shape[0]):
            raise RuntimeError(
                "Mismatch among per_frame/window/groups lengths in multiclass features: "
                f"per_frame={n_rows}, window={window_df.shape[0]}, groups={groups.shape[0]}"
            )
        for name, arr in labels_by_behavior_arr.items():
            if arr.shape[0] != n_rows:
                raise RuntimeError(
                    "Mismatch between multiclass label rows and features: "
                    f"{name} has {arr.shape[0]} rows, features have {n_rows}"
                )

        return {
            "window": window_df,
            "per_frame": per_frame_df,
            "labels_by_behavior": labels_by_behavior_arr,
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
        try:
            self._video_manager.get_cached_pose_path(vid)
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

        # In multi-class mode "None" is a reserved background class that must not
        # appear in the project's behavior list. Reject the rename here so it is
        # caught even when no multi-class classifier has been trained/saved yet.
        if (
            self._settings_manager.classifier_mode == ClassifierMode.MULTICLASS
            and new_name == MULTICLASS_NONE_BEHAVIOR
        ):
            raise ValueError(
                f"Cannot rename a behavior to the reserved multi-class name "
                f"{MULTICLASS_NONE_BEHAVIOR!r}"
            )

        # Classifier and prediction storage differs between modes, so update the
        # mode-specific artifacts separately.
        if self._settings_manager.classifier_mode == ClassifierMode.MULTICLASS:
            self._rename_behavior_multiclass_artifacts(old_name, new_name)
        else:
            self._rename_behavior_binary_artifacts(old_name, new_name)

        # rename labels inside every video's annotation file (mode-independent:
        # labels are always keyed by behavior name on disk)
        for video in self._video_manager.videos:
            if (labels := self._video_manager.load_video_labels(video)) is None:
                continue
            labels.rename_behavior(old_name, new_name)
            pose = self.load_pose_est(self._video_manager.video_path(video))
            self.save_annotations(labels, pose)

        # update project settings
        self._settings_manager.rename_behavior(old_name, new_name)

    def _rename_behavior_binary_artifacts(self, old_name: str, new_name: str) -> None:
        """Rename binary-mode classifier and prediction artifacts for a behavior.

        Binary mode stores one classifier pickle per behavior (keyed by safe
        name) and one prediction group per behavior inside each video's HDF5
        file. Both are renamed to track the behavior's new name.

        Args:
            old_name: current behavior name
            new_name: new behavior name
        """
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
            if not pred_file.exists():
                continue
            with h5py.File(pred_file, "r+") as hf:
                if "predictions" in hf and safe_old_name in hf["predictions"]:
                    grp = hf["predictions"]
                    # If dataset already exists at the destination, remove it first
                    if safe_new_name in grp:
                        del grp[safe_new_name]
                    grp.move(safe_old_name, safe_new_name)

    def _rename_behavior_multiclass_artifacts(self, old_name: str, new_name: str) -> None:
        """Rename multi-class classifier and prediction artifacts for a behavior.

        Multi-class mode stores a single classifier pickle and a single
        prediction group shared across all behaviors; the behavior name is
        recorded inside each (``MultiClassClassifier._behavior_names`` and the
        per-video ``class_names`` dataset). These are updated in place so they
        do not desync from the project settings after a rename.

        Args:
            old_name: current behavior name
            new_name: new behavior name
        """
        # Imported here rather than at module scope because jabs.classifier
        # imports jabs.project, so a top-level import would be circular.
        from jabs.classifier import MultiClassClassifier

        # update behavior name inside the saved multi-class classifier
        classifier_path = self._paths.classifier_dir / MULTICLASS_CLASSIFIER_FILENAME
        new_classifier_file: str | None = None
        new_classifier_hash: str | None = None
        if classifier_path.exists():
            classifier = MultiClassClassifier.from_pickle(classifier_path)
            if old_name in classifier.behavior_names:
                classifier.rename_behavior(old_name, new_name)
                # The pickle's contents changed, so drop the stale file identity
                # and let save() record a hash matching the rewritten file.
                classifier.reset_persistence_identity()
                classifier.save(classifier_path)
                # Capture the rewritten pickle's identity so per-video prediction
                # metadata can be repointed at the renamed classifier file.
                new_classifier_file = classifier.classifier_file
                new_classifier_hash = classifier.classifier_hash

        # update the class_names dataset inside each video's prediction group
        safe_multiclass_name = to_safe_name(MULTICLASS_PREDICTION_KEY)
        for video in self._video_manager.videos:
            pred_file = self._paths.prediction_dir / Path(video).with_suffix(".h5").name
            if not pred_file.exists():
                continue
            with h5py.File(pred_file, "r+") as hf:
                group = hf.get(f"predictions/{safe_multiclass_name}")
                if group is None or "class_names" not in group:
                    continue
                names = [
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in group["class_names"][()]
                ]
                if old_name not in names:
                    continue
                names[names.index(old_name)] = new_name
                del group["class_names"]
                group.create_dataset(
                    "class_names",
                    data=np.array(names, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                # The rename re-saved the classifier with a new content hash, so
                # repoint this group's classifier metadata at the rewritten file;
                # the predictions themselves remain valid (only a class name
                # changed). Skipped when no classifier pickle was re-saved.
                if new_classifier_file is not None:
                    group.attrs["classifier_file"] = new_classifier_file
                if new_classifier_hash is not None:
                    group.attrs["classifier_hash"] = new_classifier_hash
