#!/usr/bin/env python
"""
jabs-classify

Todo:
- use click for implementing command line interface with multiple commands
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, TextColumn
from sklearn.exceptions import InconsistentVersionWarning

from jabs.classifier import Classifier, MultiClassClassifier
from jabs.core.constants import APP_NAME, MULTICLASS_NONE_BEHAVIOR
from jabs.core.enums import CacheFormat, ClassifierType
from jabs.feature_extraction import IdentityFeatures
from jabs.pose_estimation import open_pose_file
from jabs.project.prediction_manager import MULTICLASS_PREDICTION_KEY, PredictionManager

DEFAULT_FPS = 30

# find out which classifiers are supported in this environment
__CLASSIFIER_CHOICES = Classifier().classifier_choices()


def get_pose_stem(pose_path: Path) -> str:
    """Get the stem name of a pose file.

    Takes a pose path as input and returns the name component with the
    '_pose_est_v#.h5' suffix removed.

    Args:
        pose_path: Path to the pose estimation file.

    Returns:
        Stem portion of the filename without the pose suffix.

    Raises:
        ValueError: If the path does not match the expected pose file naming convention.
    """
    m = re.match(r"^(.+)(_pose_est_v[0-9]+\.h5)$", pose_path.name)
    if m:
        return m.group(1)
    else:
        raise ValueError(f"{pose_path} is not a valid pose file path")


def _load_classifier_from_pickle(path: Path) -> Classifier | MultiClassClassifier:
    """Load a binary or multi-class classifier from a pickle file.

    Peeks at the deserialized type, then delegates to the class-specific
    ``from_pickle()`` classmethod so that version checks, supported
    classifier-type checks, and metadata backfill are applied consistently.

    Args:
        path: Path to the saved classifier pickle file.

    Returns:
        Loaded ``Classifier`` or ``MultiClassClassifier`` instance.

    Raises:
        ValueError: If the file was trained with an incompatible sklearn or JABS
            version, uses an unsupported classifier type, or contains an
            unrecognized object type.
        FileNotFoundError: If ``path`` does not exist.
        PermissionError: If the file cannot be read.
        Exception: Other exceptions raised by joblib/pickle during deserialization
            (e.g. corrupt file).
    """
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", InconsistentVersionWarning)
        obj = joblib.load(path)
        for w in caught_warnings:
            if issubclass(w.category, InconsistentVersionWarning):
                raise ValueError("Classifier trained with a different version of sklearn.")
            warnings.warn(w.message, w.category, stacklevel=2)

    if isinstance(obj, MultiClassClassifier):
        return MultiClassClassifier.from_pickle(path)
    if isinstance(obj, Classifier):
        return Classifier.from_pickle(path)
    raise ValueError(f"Unrecognized classifier type in {path}: {type(obj).__name__}")


def _is_multiclass_training_file(path: Path) -> bool:
    """Return True if the training file contains multi-class training data.

    Args:
        path: Path to the training HDF5 file.

    Returns:
        True if the file has ``classifier_mode == "multiclass"``, False otherwise.
    """
    with h5py.File(path, "r") as f:
        return f.attrs.get("classifier_mode", "") == "multiclass"


def train_multiclass(
    training_file: Path, classifier_type: ClassifierType | None = None
) -> MultiClassClassifier:
    """Train a multi-class classifier using the provided training file.

    Loads training data from the specified HDF5 file, initializes a
    ``MultiClassClassifier``, and prints training details such as behavior names,
    classifier type, window size, and other relevant settings.

    Args:
        training_file: Path to the multi-class training HDF5 file exported by JABS.
        classifier_type: Override the classifier algorithm stored in the training file.
            If ``None``, the type recorded in the file is used.

    Returns:
        Trained ``MultiClassClassifier`` instance.
    """
    classifier = MultiClassClassifier.from_training_file(
        training_file, classifier_type=classifier_type
    )
    classifier_settings = classifier.project_settings

    print("Training multi-class classifier for:", ", ".join(classifier.behavior_names))
    print(f"  Classifier Type: {classifier.classifier_name}")
    print(f"  Window Size: {classifier_settings['window_size']}")
    print(f"  Social: {classifier_settings['social']}")
    print(f"  Balanced Labels: {classifier_settings['balance_labels']}")
    print(f"  Symmetric Behavior: {classifier_settings['symmetric_behavior']}")
    print(f"  CM Units: {bool(classifier_settings['cm_units'])}")

    return classifier


def train_and_classify(
    training_file_path: Path,
    input_pose_file: Path,
    out_dir: Path,
    fps: int = DEFAULT_FPS,
    feature_dir: str | None = None,
    cache_window: bool = False,
    use_pose_hash: bool = False,
    classifier_type: ClassifierType | None = None,
) -> None:
    """Train a classifier using the provided training file and classify behaviors in a pose file.

    Loads the training data, trains a classifier, and applies it to the input pose file
    to predict behaviors. The classification results are saved to the specified output directory.

    Args:
        training_file_path: Path to the training HDF5 file.
        input_pose_file: Path to the input pose HDF5 file to classify.
        out_dir: Directory to store classification output.
        fps: Frames per second for feature extraction.
        feature_dir: Directory for feature cache. If provided, features are cached here.
        cache_window: Whether to cache window features.
        use_pose_hash: Include pose file hash as a subdirectory in the cache path.
        classifier_type: Override the classifier algorithm stored in the training file.
            If ``None``, the type recorded in the file is used.
    """
    if not training_file_path.exists():
        sys.exit("Unable to open training data\n")

    if _is_multiclass_training_file(training_file_path):
        classifier: Classifier | MultiClassClassifier = train_multiclass(
            training_file_path, classifier_type=classifier_type
        )
    else:
        classifier = train(training_file_path, classifier_type=classifier_type)
    classify_pose(
        classifier,
        input_pose_file,
        out_dir,
        classifier.behavior_name,
        fps,
        feature_dir,
        cache_window,
        use_pose_hash=use_pose_hash,
    )


def classify_pose(
    classifier: Classifier | MultiClassClassifier,
    input_pose_file: Path,
    out_dir: Path,
    behavior: str | None = None,
    fps: int = DEFAULT_FPS,
    feature_dir: str | None = None,
    cache_window: bool = False,
    use_pose_hash: bool = False,
) -> None:
    """Classify behaviors in a pose file using a trained classifier.

    Loads pose data, extracts features for each identity, predicts behavior labels
    and probabilities, and writes the results to an output HDF5 file.

    For binary classifiers, ``behavior`` names the behavior being classified and is
    used as the prediction record key. For multi-class classifiers, ``behavior`` is
    ignored - the key is always ``MULTICLASS_PREDICTION_KEY`` and ``class_names``
    are populated from the classifier.

    Args:
        classifier: Trained binary or multi-class classifier instance.
        input_pose_file: Path to the input pose HDF5 file.
        out_dir: Directory to store classification output.
        behavior: Behavior name for binary classifiers. Ignored for multi-class.
        fps: Frames per second for feature extraction.
        feature_dir: Directory for feature cache. If provided, features are cached here.
        cache_window: Whether to cache window features.
        use_pose_hash: Include pose file hash as a subdirectory in the cache path.

    Raises:
        ValueError: If a binary classifier is given but ``behavior`` is None.
    """
    multiclass = isinstance(classifier, MultiClassClassifier)

    if multiclass:
        class_names: list[str] | None = [MULTICLASS_NONE_BEHAVIOR, *classifier.behavior_names]
        behavior_key = MULTICLASS_PREDICTION_KEY
    else:
        if behavior is None:
            raise ValueError("behavior is required for binary classifiers")
        class_names = None
        behavior_key = behavior

    pose_est = open_pose_file(input_pose_file)
    pose_stem = get_pose_stem(input_pose_file)

    n_identities = pose_est.num_identities
    n_frames = pose_est.num_frames

    prediction_labels = np.full((n_identities, n_frames), -1, dtype=np.int8)
    if multiclass:
        n_classes = len(class_names)  # type: ignore[arg-type]
        prediction_prob: np.ndarray = np.zeros(
            (n_identities, n_frames, n_classes), dtype=np.float32
        )
    else:
        prediction_prob = np.zeros((n_identities, n_frames), dtype=np.float32)

    classifier_settings = classifier.project_settings

    print(f"Classifying {input_pose_file}...")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} of {task.total} identities"),
    ) as progress:
        task = progress.add_task("Processing", total=n_identities)
        for curr_id in pose_est.identities:
            features = IdentityFeatures(
                input_pose_file,
                curr_id,
                feature_dir,
                pose_est,
                fps=fps,
                op_settings=classifier_settings,
                cache_window=cache_window,
                cache_format=CacheFormat.PARQUET,
                include_pose_hash=use_pose_hash,
            ).get_features(classifier_settings["window_size"])

            per_frame_features = pd.DataFrame(features["per_frame"])
            window_features = pd.DataFrame(features["window"])
            data = classifier.combine_data(per_frame_features, window_features)

            if data.shape[0] > 0:
                prob = classifier.predict_proba(data, features["frame_indexes"])
                predictions, confidence = classifier.derive_predictions(prob)
                prediction_labels[curr_id] = predictions
                # Multiclass: persist full class-probability matrix (n_frames, n_classes).
                # Binary: persist per-frame confidence scalar.
                prediction_prob[curr_id] = prob if multiclass else confidence
            progress.update(task, advance=1)

    print(f"Writing predictions to {out_dir}")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Unable to create output directory: {e}")

    behavior_out_path = out_dir / (pose_stem + "_behavior.h5")

    PredictionManager.write_predictions(
        behavior_key,
        behavior_out_path,
        prediction_labels,
        prediction_prob,
        pose_est,
        classifier,
        class_names=class_names,
    )


def train(training_file: Path, classifier_type: ClassifierType | None = None) -> Classifier:
    """Train a binary classifier using the provided training file.

    Loads training data from the specified HDF5 file, initializes a classifier,
    and prints training details such as behavior name, classifier type, window size,
    and other relevant settings.

    Args:
        training_file: Path to the training HDF5 file exported by JABS.
        classifier_type: Override the classifier algorithm stored in the training file.
            If ``None``, the type recorded in the file is used.

    Returns:
        Trained ``Classifier`` instance.
    """
    classifier = Classifier.from_training_file(training_file, classifier_type=classifier_type)
    classifier_settings = classifier.project_settings

    print("Training classifier for:", classifier.behavior_name)
    print(f"  Classifier Type: {__CLASSIFIER_CHOICES[classifier.classifier_type]}")
    print(f"  Window Size: {classifier_settings['window_size']}")
    print(f"  Social: {classifier_settings['social']}")
    print(f"  Balanced Labels: {classifier_settings['balance_labels']}")
    print(f"  Symmetric Behavior: {classifier_settings['symmetric_behavior']}")
    print(f"  CM Units: {bool(classifier_settings['cm_units'])}")

    return classifier


def main() -> None:
    """jabs-classify entrypoint - dispatch to different main functions depending on command."""
    if len(sys.argv) < 2:
        usage_main()
    elif sys.argv[1] == "classify":
        classify_main()
    elif sys.argv[1] == "train":
        train_main()
    else:
        usage_main()


def usage_main() -> None:
    """Print usage information for the script."""
    print("usage: " + script_name() + " COMMAND COMMAND_ARGS\n", file=sys.stderr)
    print("commands:", file=sys.stderr)
    print(" classify   classify a pose file", file=sys.stderr)
    print(
        " train      train a classifier that can be used to classify multiple pose files",
        file=sys.stderr,
    )
    print(
        f"\nSee `{script_name()} COMMAND --help` for information on a specific command.",
        file=sys.stderr,
    )


def classify_main() -> None:
    """Implementation of the `jabs-classify classify` command."""
    classify_args = sys.argv[2:]

    parser = argparse.ArgumentParser(prog=f"{script_name()} classify")
    required_args = parser.add_argument_group("required arguments")

    classifier_group = parser.add_argument_group(
        "optionally override the classifier specified in the training file:\n"
        " Ignored if trained classifier passed with --classifier option.\n"
        " (the following options are mutually exclusive)"
    )
    exclusive_group = classifier_group.add_mutually_exclusive_group(required=False)
    for classifer_type, classifier_str in __CLASSIFIER_CHOICES.items():
        exclusive_group.add_argument(
            f"--{classifer_type.name.lower().replace('_', '-')}",
            action="store_const",
            const=classifer_type,
            dest="classifier_type",
            help=f"Use {classifier_str}",
        )

    source_group = parser.add_argument_group("Classifier Input (one of the following is required)")
    training_group = source_group.add_mutually_exclusive_group(required=True)
    training_group.add_argument(
        "--training", help=f"Training data h5 file exported from {APP_NAME}"
    )
    training_group.add_argument(
        "--classifier",
        help=f"Classifier file produced from the `{script_name()} train` command or saved "
        "by the JABS GUI (binary .pickle or multi-class _multiclass.pickle)",
    )

    required_args.add_argument(
        "--input-pose",
        help="input HDF5 pose file.",
        required=True,
    )
    required_args.add_argument(
        "--out-dir",
        help="directory to store classification output",
        required=True,
    )
    parser.add_argument(
        "--fps",
        help=f"frames per second, default={DEFAULT_FPS}",
        type=int,
        default=DEFAULT_FPS,
    )
    parser.add_argument(
        "--feature-dir",
        help="Feature cache dir. If present, look here for features before "
        "computing. If features need to be computed, they will be saved here.",
    )
    parser.add_argument(
        "--skip-window-cache",
        help=(
            "Default will cache all features when --feature-dir is provided. Providing this flag "
            "will only cache per-frame features, reducing cache size at the cost of needing to "
            "re-calculate window features."
        ),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use-pose-hash",
        help=(
            "Include the pose file hash as a subdirectory level in the feature cache path "
            "(e.g. <feature-dir>/<video>/<pose-hash>/<identity>). "
            "Prevents collisions when multiple pipelines share a feature cache directory "
            "and different pose files happen to share the same video name."
        ),
        default=False,
        action="store_true",
    )

    args = parser.parse_args(classify_args)

    out_dir = Path(args.out_dir)
    in_pose_path = Path(args.input_pose)

    if args.training is not None:
        train_and_classify(
            Path(args.training),
            in_pose_path,
            out_dir,
            fps=args.fps,
            feature_dir=args.feature_dir,
            cache_window=not args.skip_window_cache,
            use_pose_hash=args.use_pose_hash,
            classifier_type=args.classifier_type,
        )
    elif args.classifier is not None:
        try:
            classifier = _load_classifier_from_pickle(Path(args.classifier))
        except Exception as e:
            print(f"Unable to load classifier from {args.classifier}:")
            sys.exit(str(e))

        classifier_settings = classifier.project_settings
        print(f"Classifying using trained classifier: {args.classifier}")

        if isinstance(classifier, MultiClassClassifier):
            print("  Mode: multi-class")
            print(f"  Behaviors: {', '.join(classifier.behavior_names)}")
            print(f"  Window Size: {classifier_settings['window_size']}")
            behavior = None
        else:
            try:
                print(f"  Classifier type: {__CLASSIFIER_CHOICES[classifier.classifier_type]}")
            except KeyError:
                sys.exit("Error: Classifier type not supported on this platform")
            behavior = classifier.behavior_name
            print(f"  Behavior: {behavior}")
            print(f"  Window Size: {classifier_settings['window_size']}")
            print(f"  Social: {classifier_settings['social']}")
            print(f"  CM Units: {classifier_settings['cm_units']}")

        classify_pose(
            classifier,
            in_pose_path,
            out_dir,
            behavior,
            fps=args.fps,
            feature_dir=args.feature_dir,
            cache_window=not args.skip_window_cache,
            use_pose_hash=args.use_pose_hash,
        )


def train_main() -> None:
    """Implementation of the `jabs-classify train` command."""
    train_args = sys.argv[2:]

    parser = argparse.ArgumentParser(prog=f"{script_name()} train")
    parser.add_argument("training_file", help=f"Training h5 file exported by {APP_NAME}")
    parser.add_argument("out_file", help="output filename")

    args = parser.parse_args(train_args)
    training_path = Path(args.training_file)

    if not training_path.exists():
        sys.exit("Unable to open training data\n")

    trained: Classifier | MultiClassClassifier
    if _is_multiclass_training_file(training_path):
        trained = train_multiclass(training_path)
    else:
        trained = train(training_path)

    print(f"Saving trained classifier to '{args.out_file}'")
    trained.save(Path(args.out_file))


def script_name() -> str:
    """Return the script name."""
    return Path(sys.argv[0]).name


if __name__ == "__main__":
    main()
