#!/usr/bin/env python
"""
jabs-classify

Todo:
- use click for implementing command line interface with multiple commands
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, TextColumn

from jabs.classifier import Classifier
from jabs.constants import APP_NAME
from jabs.feature_extraction import IdentityFeatures
from jabs.pose_estimation import open_pose_file
from jabs.project.prediction_manager import PredictionManager

DEFAULT_FPS = 30

# find out which classifiers are supported in this environment
__CLASSIFIER_CHOICES = Classifier().classifier_choices()


def get_pose_stem(pose_path: Path):
    """get the stem name of a pose file

    takes a pose path as input and returns the name component with the '_pose_est_v#.h5' suffix removed
    """
    m = re.match(r"^(.+)(_pose_est_v[0-9]+\.h5)$", pose_path.name)
    if m:
        return m.group(1)
    else:
        raise ValueError(f"{pose_path} is not a valid pose file path")


def train_and_classify(
    training_file_path: Path,
    input_pose_file: Path,
    out_dir: Path,
    fps=DEFAULT_FPS,
    feature_dir: str | None = None,
    cache_window: bool = False,
):
    """Train a classifier using the provided training file and classify behaviors in a pose file.

    Loads the training data, trains a classifier, and applies it to the input pose file to predict behaviors.
    The classification results are saved to the specified output directory.

    Args:
        training_file_path (Path): Path to the training HDF5 file.
        input_pose_file (Path): Path to the input pose HDF5 file to classify.
        out_dir (Path): Directory to store classification output.
        fps (int, optional): Frames per second for feature extraction. Defaults to DEFAULT_FPS.
        feature_dir (str or None, optional): Directory for feature cache. If provided, features are cached here.
        cache_window (bool, optional): Whether to cache window features. Defaults to False.
    """
    if not training_file_path.exists():
        sys.exit("Unable to open training data\n")

    classifier = train(training_file_path)
    classify_pose(
        classifier,
        input_pose_file,
        out_dir,
        classifier.behavior_name,
        fps,
        feature_dir,
        cache_window,
    )


def classify_pose(
    classifier: Classifier,
    input_pose_file: Path,
    out_dir: Path,
    behavior: str,
    fps=DEFAULT_FPS,
    feature_dir: str | None = None,
    cache_window: bool = False,
):
    """Classify behaviors in a pose file using a trained classifier.

    Loads pose data, extracts features for each identity, predicts behavior labels and probabilities,
    and writes the results to an output HDF5 file.

    Args:
        classifier (Classifier): Trained classifier instance.
        input_pose_file (Path): Path to the input pose HDF5 file.
        out_dir (Path): Directory to store classification output.
        behavior (str): Name of the behavior being classified.
        fps (int, optional): Frames per second for feature extraction. Defaults to DEFAULT_FPS.
        feature_dir (str or None, optional): Directory for feature cache. If provided, features are cached here.
        cache_window (bool, optional): Whether to cache window features. Defaults to False.
    """
    pose_est = open_pose_file(input_pose_file)
    pose_stem = get_pose_stem(input_pose_file)

    # allocate numpy arrays to write to h5 file
    prediction_labels = np.full((pose_est.num_identities, pose_est.num_frames), -1, dtype=np.int8)
    prediction_prob = np.zeros_like(prediction_labels, dtype=np.float32)

    classifier_settings = classifier.project_settings

    print(f"Classifying {input_pose_file}...")

    # run prediction for each identity
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} of {task.total} identities"),
    ) as progress:
        task = progress.add_task("Processing", total=pose_est.num_identities)
        for curr_id in pose_est.identities:
            features = IdentityFeatures(
                input_pose_file,
                curr_id,
                feature_dir,
                pose_est,
                fps=fps,
                op_settings=classifier_settings,
                cache_window=cache_window,
            ).get_features(classifier_settings["window_size"])

            per_frame_features = pd.DataFrame(
                IdentityFeatures.merge_per_frame_features(features["per_frame"])
            )
            window_features = pd.DataFrame(
                IdentityFeatures.merge_window_features(features["window"])
            )

            data = Classifier.combine_data(per_frame_features, window_features)

            if data.shape[0] > 0:
                pred = classifier.predict(data)
                pred_prob = classifier.predict_proba(data)

                # Keep the probability for the predicted class only.
                # The following code uses some
                # numpy magic to use the pred array as column indexes
                # for each row of the pred_prob array we just computed.
                pred_prob = pred_prob[np.arange(len(pred_prob)), pred]

                # Only copy out predictions where there was a valid pose
                prediction_labels[curr_id, features["frame_indexes"]] = pred[
                    features["frame_indexes"]
                ]
                prediction_prob[curr_id, features["frame_indexes"]] = pred_prob[
                    features["frame_indexes"]
                ]
            progress.update(task, advance=1)

    print(f"Writing predictions to {out_dir}")

    behavior_out_dir = out_dir
    try:
        behavior_out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Unable to create output directory: {e}")
    behavior_out_path = behavior_out_dir / (pose_stem + "_behavior.h5")

    PredictionManager.write_predictions(
        behavior,
        behavior_out_path,
        prediction_labels,
        prediction_prob,
        pose_est,
        classifier,
    )


def train(training_file: Path) -> Classifier:
    """Train a classifier using the provided training file.

    Loads training data from the specified HDF5 file, initializes a classifier,
    and prints training details such as behavior name, classifier type, window size,
    and other relevant settings.

    Args:
        training_file (Path): Path to the training HDF5 file exported by JABS.

    Returns:
        Classifier: The trained classifier instance.
    """
    classifier = Classifier.from_training_file(training_file)
    classifier_settings = classifier.project_settings

    print("Training classifier for:", classifier.behavior_name)
    print(f"  Classifier Type: {__CLASSIFIER_CHOICES[classifier.classifier_type]}")
    print(f"  Window Size: {classifier_settings['window_size']}")
    print(f"  Social: {classifier_settings['social']}")
    print(f"  Balanced Labels: {classifier_settings['balance_labels']}")
    print(f"  Symmetric Behavior: {classifier_settings['symmetric_behavior']}")
    print(f"  CM Units: {bool(classifier_settings['cm_units'])}")

    return classifier


def main():
    """jabs-classify entrypoint. dispatch to different main functions depending on command specified"""
    if len(sys.argv) < 2:
        usage_main()
    elif sys.argv[1] == "classify":
        classify_main()
    elif sys.argv[1] == "train":
        train_main()
    else:
        usage_main()


def usage_main():
    """print usage information for the script"""
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


def classify_main():
    """implementation of the `jabs-classify classify` command"""
    # strip out the 'command' from sys.argv
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
        help=f"Classifier file produced from the `{script_name()} train` command",
    )

    required_args.add_argument(
        "--input-pose",
        help="input HDF5 pose file (v2, v3, v4, or v5).",
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
            "Default will cache all features when --feature-dir is provided. Providing this flag will only cache "
            "per-frame features, reducing cache size at the cost of needing to re-calculate window features."
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
        )
    elif args.classifier is not None:
        try:
            classifier = Classifier()
            classifier.load(Path(args.classifier))
        except ValueError as e:
            print(f"Unable to load classifier from {args.classifier}:")
            sys.exit(str(e))

        behavior = classifier.behavior_name
        classifier_settings = classifier.project_settings

        print(f"Classifying using trained classifier: {args.classifier}")
        try:
            print(f"  Classifier type: {__CLASSIFIER_CHOICES[classifier.classifier_type]}")
        except KeyError:
            sys.exit("Error: Classifier type not supported on this platform")
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
        )


def train_main():
    """implementation of the `jabs-classify train` command"""
    # strip out the 'command' component from sys.argv
    train_args = sys.argv[2:]

    parser = argparse.ArgumentParser(prog=f"{script_name()} train")
    parser.add_argument("training_file", help=f"Training h5 file exported by {APP_NAME}")
    parser.add_argument("out_file", help="output filename")

    args = parser.parse_args(train_args)
    classifier = train(args.training_file)

    print(f"Saving trained classifier to '{args.out_file}'")
    classifier.save(Path(args.out_file))


def script_name() -> str:
    """return the script name"""
    return Path(sys.argv[0]).name


if __name__ == "__main__":
    main()
