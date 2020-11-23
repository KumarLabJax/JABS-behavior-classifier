import hashlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

from src.pose_estimation import get_pose_path, open_pose_file, get_frames_from_file
from src.version import version_str
from src.video_stream.utilities import get_frame_count
from src.video_stream import VideoStream
from .video_labels import VideoLabels


class Project:
    """ represents a labeling project """
    _ROTTA_DIR = 'rotta'
    __PROJECT_SETTING_FILE = 'project_settings.json'
    __PROJECT_FILE = 'project.json'
    __DEFAULT_UMASK = 0o775

    __PREDICTION_FILE_VERSION = 1

    def __init__(self, project_path, use_cache=True):
        """
        Open a project at a given path. A project is a directory that contains
        avi files and their corresponding pose_est_v3.h5 files as well as json
        files containing project metadata and annotations.
        :param project_path: path to project directory

        TODO: catch ValueError that this might raise when opening a project
        """

        # make sure this is a pathlib.Path and not a string
        self._project_dir_path = Path(project_path)

        self._annotations_dir = (self._project_dir_path / self._ROTTA_DIR /
                                 "annotations")
        self._feature_dir = (self._project_dir_path / self._ROTTA_DIR /
                             "features")

        self._prediction_dir = (self._project_dir_path / self._ROTTA_DIR /
                                "predictions")

        self._project_file = (self._project_dir_path / self._ROTTA_DIR /
                              self.__PROJECT_FILE)

        self._classifier_dir = (self._project_dir_path / self._ROTTA_DIR /
                              'classifiers')

        if use_cache:
            self._cache_dir = (self._project_dir_path / self._ROTTA_DIR /
                               'cache')
        else:
            self._cache_dir = None

        # get list of video files in the project directory
        # TODO: we could check to see if the matching .h5 file exists
        self._videos = [f.name for f in self._project_dir_path.glob("*.avi")]
        self._videos.sort()

        # if project directory doesn't exist, create it (empty project)
        # parent directory must exist.
        Path(project_path).mkdir(mode=self.__DEFAULT_UMASK, exist_ok=True)

        # make sure the app subdirectory directory exists to store project
        # metadata and annotations
        Path(project_path, self._ROTTA_DIR).mkdir(mode=self.__DEFAULT_UMASK,
                                                  exist_ok=True)

        # make sure other app directories exist
        self._annotations_dir.mkdir(mode=self.__DEFAULT_UMASK, exist_ok=True)
        self._feature_dir.mkdir(mode=self.__DEFAULT_UMASK, exist_ok=True)
        self._prediction_dir.mkdir(mode=self.__DEFAULT_UMASK, exist_ok=True)

        if use_cache:
            self._cache_dir.mkdir(mode=self.__DEFAULT_UMASK, exist_ok=True)

        # load any saved project metadata
        self._metadata = self.load_metadata()

        # unsaved annotations
        self._unsaved_annotations = {}

        self._total_project_identities = 0

        # get list of video files in the project directory
        self._videos = self.get_videos(self._project_dir_path)
        self._videos.sort()

        err = False
        for v in self.videos:
            if self.__has_pose(v) is False:
                print(f"{v} missing pose file", file=sys.stderr)
                err = True
        if err:
            raise ValueError("Project missing pose file for one or more video")

        err = False
        for v in self.videos:
            path = get_pose_path(self.video_path(v))
            pose_frames = get_frames_from_file(path)
            vid_frames = VideoStream.get_nframes_from_file(self.video_path(v))
            if pose_frames != vid_frames:
                print(f"{v}: video and pose file have different number of frames",
                      file=sys.stderr)
                err = True
        if err:
            raise ValueError("Video and Pose File frame counts differ")

        video_metadata = self._metadata.get('video_files', {})
        for video in self._videos:
            vinfo = {}
            if video in video_metadata:
                nidentities = video_metadata[video].get('identities')
                pose_hash = video_metadata[video].get('pose_hash')
                vid_hash = video_metadata[video].get('vid_hash')
                vinfo = video_metadata[video]
            else:
                nidentities = None
                pose_hash = None
                vid_hash = None

            # if the number of identities is not cached in the project metadata,
            # open the pose file to get it
            if nidentities is None:
                # this will raise a ValueError if the video does not have a
                # corresponding pose file.
                pose_file = open_pose_file(
                    get_pose_path(self.video_path(video)), self._cache_dir)
                nidentities = pose_file.num_identities
                vinfo['identities'] = nidentities
            if pose_hash is None:
                vinfo['pose_hash'] = self.__hash_file(
                    get_pose_path(self.video_path(video)))
            if vid_hash is None:
                vinfo['vid_hash'] = self.__hash_file(self.video_path(video))

            self._total_project_identities += nidentities
            video_metadata[video] = vinfo
        self.save_metadata({'video_files': video_metadata})

        # determine if this project relies on social features or not
        self._has_social_features = False
        for i, vid in enumerate(self._videos):
            vid_path = self.video_path(vid)
            pose_path = get_pose_path(vid_path)
            curr_has_social = pose_path.name.endswith('v3.h5')

            if i == 0:
                self._has_social_features = curr_has_social
            else:
                # here we're just making sure everything is consistent,
                # otherwise we throw a ValueError
                if curr_has_social != self._has_social_features:
                    raise ValueError('Found a pose estimation mismatch in project')

    @property
    def videos(self):
        """
        get list of video files that are in this project directory
        :return: list of file names (file names only, without path)
        """
        return self._videos

    @property
    def feature_dir(self):
        return self._feature_dir

    @property
    def annotation_dir(self):
        return self._annotations_dir

    @property
    def has_social_features(self):
        return self._has_social_features

    @property
    def metadata(self):
        """
        get the project metadata and preferences.

        Returns a copy of the metadata dict, so that self._info can't be
        modified
        """
        return dict(self._metadata)

    @property
    def classifier_dir(self):
        return self._classifier_dir

    def load_annotation_track(self, video_name, leave_cached=False):
        """
        load an annotation track from the project directory or from a cached of
        annotations that have previously been opened and not yet saved
        :param video_name: filename of the video: string or pathlib.Path
        :param leave_cached: indicates if the VideoLabels object should be
        removed from the cache if it is found there. This should be false when
        switching active videos, but false when getting labels for training or
        classification
        :return: initialized VideoLabels object
        """

        video_filename = Path(video_name).name
        self.check_video_name(video_filename)

        path = self._annotations_dir / Path(video_filename).with_suffix('.json')

        # if this has already been opened
        if video_filename in self._unsaved_annotations:
            if leave_cached:
                annotations = self._unsaved_annotations[video_filename]
            else:
                annotations = self._unsaved_annotations.pop(video_filename)
            return VideoLabels.load(annotations)

        # if annotations already exist for this video file in the project open
        # it, otherwise create a new empty VideoLabels
        if path.exists():
            with path.open() as f:
                return VideoLabels.load(json.load(f))
        else:
            video_path = self._project_dir_path / video_filename
            nframes = get_frame_count(str(video_path))
            return VideoLabels(video_filename, nframes)

    @staticmethod
    def _to_safe_name(behavior: str):
        """
        Create a version of the given behavior name that
        should be safe to use in filenames.
        :param behavior: string behavior name
        """
        safe_behavior = re.sub('[^0-9a-zA-Z]+', '_', behavior).rstrip('_')
        # get rid of consecutive underscores
        safe_behavior = re.sub('_{2,}', '_', safe_behavior)
        return safe_behavior

    def load_pose_est(self, video_path: Path):
        """
        return a PoseEstimation object for a given video path
        :param video_path: pathlib.Path containing location of video file
        :return: PoseEstimation object (PoseEstimationV2 or PoseEstimationV3)
        :raises ValueError: if video no in project or it does not have post file
        """
        # ensure this video path is for a valid project video
        video_filename = Path(video_path).name
        self.check_video_name(video_filename)

        return open_pose_file(get_pose_path(video_path), self._cache_dir)

    def check_video_name(self, video_filename):
        """
        make sure the video name actually matches one in the project, this
        function will raise a ValueError if the video name is not valid,
        otherwise the function has no effect
        :param video_filename:
        :return: None
        :raises: ValueError if the filename is not a valid video in this project
        """
        if video_filename not in self._videos:
            raise ValueError(f"{video_filename} not in project")

    def cache_annotations(self, annotations: VideoLabels):
        """
        Cache a VideoLabels object after encoding as a JSON serializable dict.
        Used when user switches from one video to another during a labeling
        project.
        :param annotations: VideoLabels object
        :return: None
        """
        self._unsaved_annotations[annotations.filename] = annotations.as_dict()

    def save_annotations(self, annotations: VideoLabels):
        """
        save state of a VideoLabels object to the project directory
        :param annotations: VideoLabels object
        :return: None
        """
        path = self._annotations_dir / Path(
            annotations.filename).with_suffix('.json')

        with path.open(mode='w', newline='\n') as f:
            json.dump(annotations.as_dict(), f, indent=2)

        # update app version saved in project metadata if necessary
        self.__update_version()

    def save_metadata(self, data: dict):
        """
        save project settings and metadata into the project directory. This may
        include things like custom behavior labels added by the user as well
        as the most recently selected behavior label

        any keys in the project metadata dict not included in the data, will
        not be modified
        :param data: dictionary with state information to save
        :return: None
        """

        # merge data with current metadata
        self._metadata.update(data)
        self._metadata['version'] = version_str()

        # save combined info to file
        with self._project_file.open(mode='w', newline='\n') as f:
            json.dump(self._metadata, f, indent=2, sort_keys=True)

    def load_metadata(self):
        """
        load project metadata
        :return: dictionary of project metadata, empty dict if unable to open
        file (such as when the prject is first created and the file does not
        exist)
        """
        try:
            with self._project_file.open(mode='r', newline='\n') as f:
                settings = json.load(f)
        except:
            settings = {}

        return settings

    def save_cached_annotations(self):
        """
        save VideoLabel objects that have been cached
        :return: None
        """
        for video in self._unsaved_annotations:
            path = self._annotations_dir / Path(video).with_suffix('.json')

            with path.open(mode='w', newline='\n') as f:
                json.dump(self._unsaved_annotations[video], f, indent=2)

        # update app version saved in project metadata if necessary
        self.__update_version()

    def save_classifier(self, classifier, behavior: str):
        """
        Save the classifier for the given behavior
        :param classifier: the classifier to save
        :param behavior: string behavior name. This affects the path we save to
        """
        self._classifier_dir.mkdir(parents=True, exist_ok=True)
        classifier.save_classifier(
            self._classifier_dir / (self._to_safe_name(behavior) + '.pickle')
        )

        # update app version saved in project metadata if necessary
        self.__update_version()

    def load_classifier(self, classifier, behavior: str):
        """
        Save the classifier for the given behavior
        :param classifier: the classifier to load
        :param behavior: string behavior name. This affects the path we save to
        :return: True if load is successful and False if the file doesn't exist
        """
        classifier_path = (
            self._classifier_dir / (self._to_safe_name(behavior) + '.pickle')
        )
        try:
            classifier.load_classifier(classifier_path)
            return True
        except IOError:
            return False

    def save_predictions(self, predictions, probabilities,
                         frame_indexes, behavior: str):
        """
        save predictions for the current project
        :param predictions: predictions for all videos in project (dictionary
        with each video name as a key and a numpy array (#identities, #frames))
        :param probabilities: corresponding prediction probabilities, similar
        structure to predictions parameter but with floating point values
        :param frame_indexes: mapping of the predictions to video frames
        :param behavior: string behavior name

        Because the classifier does not run on every frame for every identity
        (since an identity may not exist for every frame), we extract just
        the features for the frames we need to classify. Now we want to map
        these back to the corresponding frame.
        predictions[video_name][identity, index] and
        probabilities[video_name][identity, index] correspond to the frame
        specified by frame_indexes[video][identity, index]
        """

        for video in self._videos:
            # setup an ouptut filename based on the behavior and video names
            file_base = Path(video).with_suffix('').name + ".h5"
            output_path = self._prediction_dir / self._to_safe_name(
                behavior) / file_base

            # make sure behavior directory exists
            output_path.parent.mkdir(exist_ok=True)

            # we need some info from the PoseEstimation and VideoLabels objects
            # associated with this video
            video_tracks = self.load_annotation_track(video, leave_cached=True)
            poses = open_pose_file(get_pose_path(self.video_path(video)), self._cache_dir)

            # allocate numpy arrays to write to h5 file
            prediction_labels = np.full(
                (poses.num_identities, video_tracks.num_frames), -1,
                dtype=np.int8)
            prediction_prob = np.zeros_like(prediction_labels, dtype=np.float32)

            # populate numpy arrays
            for identity in predictions[video]:
                identity_index = int(identity)

                inferred_indexes = frame_indexes[video][identity]
                track = video_tracks.get_track_labels(identity, behavior)
                manual_labels = track.get_labels()

                prediction_labels[identity_index, inferred_indexes] = predictions[video][identity]
                prediction_prob[identity_index, inferred_indexes] = probabilities[video][identity]

                # manual labels are saved with the predictions, but the
                # probability is set to -1 to indicate that the class was
                # manually assigned by the user and not inferred
                prediction_labels[identity_index,
                    manual_labels == track.Label.NOT_BEHAVIOR] = track.Label.NOT_BEHAVIOR
                prediction_prob[identity_index, manual_labels == track.Label.NOT_BEHAVIOR] = -1.0
                prediction_labels[identity_index,
                    manual_labels == track.Label.BEHAVIOR] = track.Label.BEHAVIOR
                prediction_prob[identity_index, manual_labels == track.Label.BEHAVIOR] = -1.0

            # write to h5 file
            # TODO catch exceptions
            with h5py.File(output_path, 'w') as h5:
                h5.attrs['version'] = self.__PREDICTION_FILE_VERSION
                group = h5.create_group('predictions')
                group.create_dataset('predicted_class', data=prediction_labels)
                group.create_dataset('probabilities', data=prediction_prob)
                group.create_dataset('identity_to_track', data=poses.identity_to_track)

        # update app version saved in project metadata if necessary
        self.__update_version()

    def load_predictions(self, behavior: str):
        """
        load previously saved predictions if they are present
        :param behavior: behavior to load predictions for
        :return: tuple of three dicts. For each dict, the first key is the video
         name. The value for that key is another dict with identities as keys.
         the predictions dict stores the inferred classes for each identity,
         the probabilities dict stores the probabilities of the inferences
         the frame_indexes dict stores the frame indexes the inferences
         correspond to
        """
        predictions = {}
        probabilities = {}
        frame_indexes = {}
        for video in self._videos:
            file_base = Path(video).with_suffix('').name + ".h5"
            path = self._prediction_dir / self._to_safe_name(behavior) / file_base

            nident = self._metadata['video_files'][video]['identities']

            try:
                with h5py.File(path, 'r') as h5:
                    assert h5.attrs['version'] == self.__PREDICTION_FILE_VERSION
                    group = h5['predictions']
                    assert group['predicted_class'].shape[0] == nident
                    assert group['probabilities'].shape[0] == nident
                    predictions[video] = {}
                    probabilities[video] = {}
                    frame_indexes[video] = {}
                    for i in range(nident):
                        identity = str(i)
                        indexes = np.asarray(range(group['predicted_class'].shape[1]))

                        # first, exclude any probability of -1 as that indicates
                        # a user label, not a inferred class
                        classes = group['predicted_class'][i, group['probabilities'][i] != -1.0]
                        prob = group['probabilities'][i, group['probabilities'][i] != -1.0]
                        indexes = indexes[group['probabilities'][i] != -1]

                        # now excludes a class of -1 as that indicates the
                        # identity isn't present
                        prob = prob[classes != -1]
                        indexes = indexes[classes != -1]
                        classes = classes[classes != -1]

                        # we're left with classes/probabilities for frames that
                        # were inferred and their frame indexes
                        predictions[video][identity] = classes
                        probabilities[video][identity] = prob
                        frame_indexes[video][identity] = indexes

            except IOError:
                # no saved predictions for this video
                pass
            except (AssertionError, KeyError) as e:
                print(f"unable to open saved inferences for {video}", file=sys.stderr)

        return predictions, probabilities, frame_indexes

    def video_path(self, video_file):
        """ take a video file name and generate the path used to open it """
        return Path(self._project_dir_path, video_file)

    def _read_counts(self, video, behavior):
        """
        read labeled frame and bout counts from json file
        :return: list of labeled frame and bout counts for each identity for the
        specified behavior. Each element in the list is a tuple of the form
        (
            identity,
            (behavior frame count, not behavior frame count)
            (behavior bout count, not behavior bout count)
        )
        """
        video_filename = Path(video).name
        path = self._annotations_dir / Path(video_filename).with_suffix('.json')

        counts = []

        if path.exists():
            with path.open() as f:
                labels = json.load(f).get('labels')
                for identity in labels:
                    blocks = labels[identity].get(behavior, [])
                    frames_behavior = 0
                    frames_not_behavior = 0
                    bouts_behavior = 0
                    bouts_not_behavior = 0
                    for b in blocks:
                        if b['present']:
                            bouts_behavior += 1
                            frames_behavior += b['end'] - b['start'] + 1
                        else:
                            bouts_not_behavior += 1
                            frames_not_behavior += b['end'] - b['start'] + 1

                    counts.append((identity,
                                   (frames_behavior, frames_not_behavior),
                                   (bouts_behavior, bouts_not_behavior)))
        return counts

    def counts(self, behavior):
        """
        get the labeled frame counts and bout counts for each video in the
        project
        :return: dict where keys are video names and values are lists of
        (
            identity,
            (behavior frame count, not behavior frame count),
            (behavior bout count, not behavior bout count)
        )
        """
        counts = {}
        for video in self._videos:
            counts[video] = self._read_counts(video, behavior)
        return counts

    def label_counts(self, behavior):
        """
        get counts of number of frames with labels for a behavior across
        entire project
        :return: dict where keys are video names and values are lists of
        (identity, (behavior count, not behavior count)
        """
        counts = {}
        for video in self._videos:
            video_track = self.load_annotation_track(video, leave_cached=True)
            counts[video] = video_track.label_counts(behavior)
        return counts

    def bout_counts(self, behavior):
        """
        get counts of number of frames with labels for a behavior across
        entire project
        :return: dict where keys are video names and values are lists of
        (identity, (behavior bout count, not behavior bout count) tuples
        """
        counts = {}
        for video in self._videos:
            video_track = self.load_annotation_track(video, leave_cached=True)
            counts[video] = video_track.bout_counts(behavior)
        return counts

    @property
    def total_project_identities(self):
        """
        sum the number of instances across all videos in the project
        :return: integer sum
        """
        return self._total_project_identities

    def __update_version(self):
        """ update the version number saved in project metadata """
        # only update if the version in the metadata is different from current
        version = self._metadata.get('version')
        if version != version_str():
            self.save_metadata({'version': version_str()})

    @staticmethod
    def __hash_file(file: Path):
        """ return hash """
        chunk_size = 8192
        with file.open('rb') as f:
            h = hashlib.blake2b(digest_size=20)
            c = f.read(chunk_size)
            while c:
                h.update(c)
                c = f.read(chunk_size)
        return h.hexdigest()

    @staticmethod
    def get_videos(dir_path: Path):
        """ Get list of video filenames (without path) in a directory """
        return [f.name for f in dir_path.glob("*.avi")]

    def __has_pose(self, vid: str):
        """ check to see if a video has a corresponding pose file """
        path = self._project_dir_path / vid

        try:
            get_pose_path(path)
        except ValueError:
            return False
        return True
