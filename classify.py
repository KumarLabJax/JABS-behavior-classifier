#!/usr/bin/env python

import argparse
import re
import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd

from src import APP_NAME
from src.classifier import Classifier, ClassifierType
from src.cli import cli_progress_bar
from src.feature_extraction.features import IdentityFeatures
from src.pose_estimation import open_pose_file
from src.project import Project, load_training_data, ProjectDistanceUnit

DEFAULT_FPS = 30

# find out which classifiers are supported in this environment
__CLASSIFIER_CHOICES = Classifier().classifier_choices()


def get_pose_stem(pose_path: Path):
    """
    takes a pose path as input and returns the name component
    with the '_pose_est_v#.h5' suffix removed
    """
    m = re.match(r'^(.+)(_pose_est_v[0-9]+\.h5)$', pose_path.name)
    if m:
        return m.group(1)
    else:
        raise ValueError(f"{pose_path} is not a valid pose file path")


def train_and_classify(
        training_file_path: Path,
        input_pose_file: Path,
        out_dir: Path,
        override_classifier: typing.Optional[ClassifierType] = None,
        fps=DEFAULT_FPS,
        feature_dir: typing.Optional[str] = None):
    if not training_file_path.exists():
        sys.exit(f"Unable to open training data\n")

    classifier = train(training_file_path, override_classifier)
    classify_pose(classifier, input_pose_file, out_dir, behavior, fps, feature_dir)


def classify_pose(classifier: Classifier, input_pose_file: Path, out_dir: Path,
                  behavior: str, fps=DEFAULT_FPS,
                  feature_dir: typing.Optional[str] = None):
    pose_est = open_pose_file(input_pose_file)
    pose_stem = get_pose_stem(input_pose_file)

    # allocate numpy arrays to write to h5 file
    prediction_labels = np.full(
        (pose_est.num_identities, pose_est.num_frames), -1,
        dtype=np.int8)
    prediction_prob = np.zeros_like(prediction_labels, dtype=np.float32)

    classifier_settings = classifier.project_settings

    print(f"Classifying {input_pose_file}...")

    # run prediction for each identity
    for curr_id in pose_est.identities:
        cli_progress_bar(curr_id, len(pose_est.identities),
                         complete_as_percent=False, suffix='identities')

        features = IdentityFeatures(
            input_pose_file, curr_id, feature_dir, pose_est, fps=fps, op_settings=classifier_settings
        ).get_features(classifier_settings['window_size'])
        per_frame_features = pd.DataFrame(IdentityFeatures.merge_per_frame_features(features['per_frame']))
        window_features = pd.DataFrame(IdentityFeatures.merge_window_features(features['window']))

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
            prediction_labels[curr_id, features['frame_indexes']] = pred[features['frame_indexes']]
            prediction_prob[curr_id, features['frame_indexes']] = pred_prob[features['frame_indexes']]
    cli_progress_bar(len(pose_est.identities), len(pose_est.identities),
                     complete_as_percent=False, suffix='identities')

    print(f"Writing predictions to {out_dir}")

    behavior_out_dir = out_dir / Project.to_safe_name(behavior)
    try:
        behavior_out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Unable to create output directory: {e}")
    behavior_out_path = behavior_out_dir / (pose_stem + '.h5')

    Project.write_predictions(
        behavior_out_path,
        prediction_labels,
        prediction_prob,
        pose_est
    )


def train(
        training_file: Path,
        override_classifier: typing.Optional[ClassifierType] = None
) -> Classifier:

    try:
        loaded_training_data, _ = load_training_data(training_file)
    except OSError as e:
        sys.exit(f"Unable to open training data\n{e}")

    behavior = loaded_training_data['behavior']

    classifier = Classifier()
    classifier.set_dict_settings(loaded_training_data['settings'])

    # Override the classifier type
    if override_classifier is not None:
        classifier_type = override_classifier
    else:
        classifier_type = ClassifierType(
            loaded_training_data['classifier_type'])

    if classifier_type in classifier.classifier_choices():
        classifier.set_classifier(classifier_type)
    else:
        print(f"Specified classifier type ({classifier_type.name}) "
              "is unavailable, using default "
              f"({classifier.classifier_type.name})")

    print("Training classifier for:", behavior)
    print("  Classifier Type: "
          f"{__CLASSIFIER_CHOICES[classifier.classifier_type]}")
    print(f"  Window Size: {loaded_training_data['settings']['window_size']}")
    print(f"  Social: {loaded_training_data['settings']['social']}")
    print(f"  Balanced Labels: {loaded_training_data['settings']['balance_labels']}")
    print(f"  Symmetric Behavior: {loaded_training_data['settings']['symmetric_behavior']}")
    print(f"  CM Units: {loaded_training_data['settings']['cm_units']}")

    training_features = classifier.combine_data(loaded_training_data['per_frame'],
                                                loaded_training_data['window'])
    classifier.train(
        {
            'training_data': training_features,
            'training_labels': loaded_training_data['labels']
        },
        behavior,
        random_seed=loaded_training_data['training_seed']
    )

    return classifier


def main():
    if len(sys.argv) < 2:
        usage_main()
    elif sys.argv[1] == 'classify':
        classify_main()
    elif sys.argv[1] == 'train':
        train_main()
    else:
        usage_main()


def usage_main():
    print("usage: " + script_name() + " COMMAND COMMAND_ARGS\n",
          file=sys.stderr)
    print("commands:", file=sys.stderr)
    print(" classify   classify a pose file", file=sys.stderr)
    print(" train      train a classifier that can be used to classify "
          "multiple pose files", file=sys.stderr)
    print(f"\nSee `{script_name()} COMMAND --help` for information on a "
          "specific command.", file=sys.stderr)


def classify_main():

    # strip out the 'command' from sys.argv
    classify_args = sys.argv[2:]

    parser = argparse.ArgumentParser(prog=f"{script_name()} classify")
    required_args = parser.add_argument_group("required arguments")

    classifier_group = parser.add_argument_group(
        "optionally override the classifier specified in the training file:\n"
        " Ignored if trained classifier passed with --classifier option.\n"
        " (the following options are mutually exclusive)")
    exclusive_group = classifier_group.add_mutually_exclusive_group(
        required=False)
    for classifer_type, classifier_str in __CLASSIFIER_CHOICES.items():
        exclusive_group.add_argument(
            f"--{classifer_type.name.lower().replace('_', '-')}",
            action='store_const', const=classifer_type,
            dest='classifier_type', help=f"Use {classifier_str}"
        )

    source_group = parser.add_argument_group(
        "Classifier Input (one of the following is required)")
    training_group = source_group.add_mutually_exclusive_group(required=True)
    training_group.add_argument(
        '--training', help=f'Training data h5 file exported from {APP_NAME}')
    training_group.add_argument(
        '--classifier',
        help=f'Classifier file produced from the `{script_name()} train` command')

    required_args.add_argument(
        '--input-pose',
        help='input HDF5 pose file (v2, v3, v4, or v5).',
        required=True,
    )
    required_args.add_argument(
        '--out-dir',
        help='directory to store classification output',
        required=True,
    )
    parser.add_argument(
        '--fps',
        help=f"frames per second, default={DEFAULT_FPS}",
        type=int,
        default=DEFAULT_FPS
    )
    parser.add_argument(
        '--feature-dir',
        help="Feature cache dir. If present, look here for features before "
        "computing. If features need to be computed, they will be saved here."
    )

    args = parser.parse_args(classify_args)

    out_dir = Path(args.out_dir)
    in_pose_path = Path(args.input_pose)

    if args.training is not None:
        train_and_classify(Path(args.training), in_pose_path, out_dir,
                           override_classifier=args.classifier,
                           fps=args.fps, feature_dir=args.feature_dir)
    elif args.classifier is not None:

        try:
            classifier = Classifier()
            classifier.load(Path(args.classifier))
        except ValueError as e:
            print(f"Unable to load classifier from {args.classifier}:")
            sys.exit(e)

        behavior = classifier.behavior_name
        classifier_settings = classifier.project_settings

        print(f"Classifying using trained classifier: {args.classifier}")
        try:
            print(
                f"  Classifier type: {__CLASSIFIER_CHOICES[classifier.classifier_type]}")
        except KeyError:
            sys.exit("Error: Classifier type not supported on this platform")
        print(f"  Behavior: {behavior}")
        print(f"  Window Size: {classifier_settings['window_size']}")
        print(f"  Social: {classifier_settings['social']}")
        print(f"  CM Units: {classifier_settings['cm_units']}")

        classify_pose(classifier, in_pose_path, out_dir, behavior, fps=args.fps, feature_dir=args.feature_dir)


def train_main():
    # strip out the 'command' component from sys.argv
    train_args = sys.argv[2:]

    parser = argparse.ArgumentParser(prog=f"{script_name()} train")
    classifier_group = parser.add_argument_group(
        "optionally override the classifier specified in the training file:\n"
        " (the following options are mutually exclusive)")
    exclusive_group = classifier_group.add_mutually_exclusive_group(
        required=False)
    for classifer_type, classifier_str in __CLASSIFIER_CHOICES.items():
        exclusive_group.add_argument(
            f"--{classifer_type.name.lower().replace('_', '-')}",
            action='store_const', const=classifer_type,
            dest='classifier', help=f"Use {classifier_str}"
        )
    parser.add_argument('training_file',
                        help=f"Training h5 file exported by {APP_NAME}")
    parser.add_argument('out_file',
                        help="output filename")

    args = parser.parse_args(train_args)
    classifier = train(args.training_file, args.classifier)

    print(f"Saving trained classifier to '{args.out_file}'")
    classifier.save(Path(args.out_file))


def script_name():
    return Path(sys.argv[0]).name


if __name__ == "__main__":
    main()
