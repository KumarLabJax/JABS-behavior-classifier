import contextlib
import getpass
import gzip
import json
import shutil
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import jabs.feature_extraction as fe
from jabs.pose_estimation import PoseEstimation, get_pose_path, open_pose_file
from jabs.types import ProjectDistanceUnit
from jabs.video_reader.utilities import get_fps

from .feature_manager import FeatureManager
from .prediction_manager import PredictionManager
from .project_paths import ProjectPaths
from .project_utils import to_safe_name
from .settings_manager import SettingsManager
from .track_labels import TrackLabels
from .video_labels import VideoLabels
from .video_manager import VideoManager


class Project:
    """Represents a JABS project, managing all data, settings, and operations for a project directory.

    A project consists of video files, pose files, metadata, annotations, classifier data, and possibly predictions.
    This class provides methods to access and manage project resources, including loading and saving annotations,
    managing features and predictions, archiving behaviors, and retrieving project settings.

    Args:
        project_path: Path to the project directory.
        use_cache (bool, optional): Whether to use cached data. Defaults to True.
        enable_video_check (bool, optional): Whether to check for video file validity. Defaults to True.

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

    def __init__(self, project_path, use_cache=True, enable_video_check=True):
        self._paths = ProjectPaths(Path(project_path), use_cache=use_cache)
        self._paths.create_directories()
        self._total_project_identities = 0
        self._enabled_extended_features = {}

        self._settings_manager = SettingsManager(self._paths)
        self._video_manager = VideoManager(self._paths, self._settings_manager, enable_video_check)
        self._feature_manager = FeatureManager(self._paths, self._video_manager.videos)
        self._prediction_manager = PredictionManager(self)

        # write out the defaults to the project file
        self._settings_manager.save_project_file({"defaults": self.get_project_defaults()})

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
    def labeler(self) -> str:
        """return name of labeler

        For now, this is just the username of the user running JABS.
        """
        return getpass.getuser()

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

        annotations = annotations.as_dict(pose)
        annotations["labeler"] = self.labeler

        with path.open(mode="w", newline="\n") as f:
            json.dump(annotations, f, indent=2)

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
        self, predictions, probabilities, frame_indexes, behavior: str, classifier
    ):
        """save predictions for the current project

        Args:
            predictions: predictions for all videos in project
                (dictionary with each video name as a key and a numpy array (#identities, #frames))
            probabilities: corresponding prediction probabilities,
                similar structure to predictions parameter but with floating point values
            frame_indexes: mapping of the predictions to video frames
            behavior: string behavior name
            classifier: Classifier object used to generate the
                predictions

        Note:
            Because the classifier does not run on every frame for every identity
            (since an identity may not exist for every frame), we extract just
            the features for the frames we need to classify. Now we want to map
            these back to the corresponding frame.
            predictions[video_name][identity, index] and
            probabilities[video_name][identity, index] correspond to the frame
            specified by frame_indexes[video][identity, index]
        """
        for video in self._video_manager.videos:
            # setup an output filename based on the behavior and video names
            file_base = Path(video).with_suffix("").name + ".h5"
            output_path = self._paths.prediction_dir / file_base

            # make sure behavior directory exists
            output_path.parent.mkdir(exist_ok=True)

            # we need some info from the PoseEstimation and VideoLabels objects
            # associated with this video
            poses = open_pose_file(
                get_pose_path(self._video_manager.video_path(video)),
                self._paths.cache_dir,
            )

            # allocate numpy arrays to write to h5 file
            prediction_labels = np.full(
                (poses.num_identities, poses.num_frames), -1, dtype=np.int8
            )
            prediction_prob = np.zeros_like(prediction_labels, dtype=np.float32)

            # populate numpy arrays
            for identity in predictions[video]:
                inferred_indexes = frame_indexes[video][identity]

                prediction_labels[identity, inferred_indexes] = predictions[video][identity][
                    inferred_indexes
                ]
                prediction_prob[identity, inferred_indexes] = probabilities[video][identity][
                    inferred_indexes
                ]

            # write to h5 file
            self._prediction_manager.write_predictions(
                behavior,
                output_path,
                prediction_labels,
                prediction_prob,
                poses,
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

        # archive labels
        archived_labels = {}
        for video in self._video_manager.videos:
            labels = self._video_manager.load_video_labels(video)

            # if no labels for video skip it
            if labels is None:
                continue

            pose = self.load_pose_est(self._video_manager.video_path(video))
            annotations = labels.as_dict(pose)
            for ident in annotations["labels"]:
                if behavior in annotations["labels"][ident]:
                    if video not in archived_labels:
                        archived_labels[video] = {"num_frames": annotations["num_frames"]}
                        archived_labels[video][behavior] = {}
                    archived_labels[video][behavior][ident] = annotations["labels"][ident].pop(
                        behavior
                    )
            self.save_annotations(VideoLabels.load(annotations), pose)

        # write the archived labels out
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with gzip.open(self._paths.archive_dir / f"{safe_behavior}_{ts}.json.gz", "wt") as f:
            json.dump(archived_labels, f, indent=True)

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
            counts[video] = self.read_counts(video, behavior)
        return counts

    def get_labeled_features(
        self,
        behavior: str | None = None,
        progress_callable: Callable[[], None] | None = None,
        should_terminate_callable: Callable[[], None] | None = None,
    ) -> tuple[dict, dict]:
        """the features for all labeled frames

        NOTE: this will currently take a very long time to run if the features
        have not already been computed

        Args:
            behavior: the behavior settings to get labeled features for
                if None, will use project defaults (all available features)
            progress_callable: if provided this will be called
                with no args every time an identity is processed to facilitate
                progress tracking
            should_terminate_callable: if provided this will be called to check if
                the user has requested to terminate the operation. This callable
                should raise a ThreadTerminatedError if the user has requested
                early termination.

        Returns:
            two dicts: features, group_mappings

            The first dict contains features for all labeled frames and has the
            following keys:

            {
                'window': ,
                'per_frame': ,
                'labels': ,
                'groups': ,
            }

            The values contained in the first dict are suitable to pass as
            arguments to the Classifier.leave_one_group_out() method.

            The second dict in the tuple has group ids as the keys, and the
            values are a dict containing the video and identity that corresponds to
            that group id:

            {
              <group id>: {'video': <video filename>, 'identity': <identity},
              ...
            }
        """
        all_per_frame = []
        all_window = []
        all_labels = []
        all_groups = []
        group_mapping = {}

        group_id = 0
        for video in self._video_manager.videos:
            # check if early termination is requested
            if should_terminate_callable:
                should_terminate_callable()

            video_labels = self._video_manager.load_video_labels(video)

            # if there are no labels for this video, skip it
            if video_labels is None:
                if progress_callable is not None:
                    # increment progress bar for each skipped identity in the video
                    for _ in range(self._video_manager.get_video_identity_count(video)):
                        progress_callable()
                continue

            video_path = self._video_manager.video_path(video)
            pose_est = self.load_pose_est(video_path)
            # fps used to scale some features from per pixel time unit to
            # per second
            fps = get_fps(str(video_path))

            for identity in pose_est.identities:
                # check if early termination is requested
                if should_terminate_callable:
                    should_terminate_callable()

                group_mapping[group_id] = {"video": video, "identity": identity}

                labels = video_labels.get_track_labels(str(identity), behavior).get_labels()

                # because we're allowing the user to label frames where the identity drops out,
                # we need to exclude labels where the identity does not exist
                # copy labels array to avoid side effect
                labels = labels.copy()
                labels[pose_est.identity_mask(identity) == 0] = TrackLabels.Label.NONE

                # if there are no labels for this identity, skip it
                if (
                    np.count_nonzero(
                        (labels == TrackLabels.Label.BEHAVIOR)
                        | (labels == TrackLabels.Label.NOT_BEHAVIOR)
                    )
                    == 0
                ):
                    if progress_callable is not None:
                        progress_callable()
                    continue

                features = fe.IdentityFeatures(
                    video,
                    identity,
                    self.feature_dir,
                    pose_est,
                    fps=fps,
                    op_settings=self._settings_manager.get_behavior(behavior),
                )

                per_frame_features = features.get_per_frame(labels)
                per_frame_features = fe.IdentityFeatures.merge_per_frame_features(
                    per_frame_features
                )
                per_frame_features = pd.DataFrame(per_frame_features)
                all_per_frame.append(per_frame_features)

                window_features = features.get_window_features(
                    self._settings_manager.get_behavior(behavior)["window_size"], labels
                )
                window_features = fe.IdentityFeatures.merge_window_features(window_features)
                window_features = pd.DataFrame(window_features)
                all_window.append(window_features)

                all_labels.append(labels[labels != TrackLabels.Label.NONE])

                all_groups.append(np.full(per_frame_features.shape[0], group_id))
                group_id += 1

                if progress_callable is not None:
                    progress_callable()

        return {
            "window": pd.concat(all_window, join="inner"),
            "per_frame": pd.concat(all_per_frame, join="inner"),
            "labels": np.concatenate(all_labels),
            "groups": np.concatenate(all_groups),
        }, group_mapping

    def __has_pose(self, vid: str):
        """check to see if a video has a corresponding pose file"""
        path = self._paths.project_dir / vid

        try:
            get_pose_path(path)
        except ValueError:
            return False
        return True

    def read_counts(self, video, behavior) -> list[tuple]:
        """read labeled frame and bout counts from json file

        Returns:
            list of labeled frame and bout counts for each identity for
            the specified behavior. Each element in the list is a tuple of the form
            (
                identity,
                (fragmented behavior frame count, fragmented not behavior frame count),
                (fragmented behavior bout count, fragmented not behavior bout count),
                (unfragmented behavior frame count, unfragmented not behavior frame count)
                (unfragmented behavior bout count, unfragmented not behavior bout count)
            )

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
        counts = []

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

                    counts.append(
                        (
                            identity,
                            fragmented_counts[0],
                            fragmented_counts[1],
                            unfragmented_counts[0],
                            unfragmented_counts[1],
                        )
                    )
        return counts
