import argparse
import re
import sys
import typing
from pathlib import Path

import h5py
import numpy as np

from src import APP_NAME
from src.cli import cli_progress_bar
from src.classifier import Classifier
from src.feature_extraction.features import IdentityFeatures
from src.pose_estimation import open_pose_file
from src.project import Project, load_training_data


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


def classify_pose(training_file: Path, input_pose_file: Path, out_dir: Path,
                  override_classifier: typing.Optional[Classifier.ClassifierType]=None):
    pose_est = open_pose_file(input_pose_file)
    pose_stem = get_pose_stem(input_pose_file)

    try:
        training_file, _ = load_training_data(training_file)
    except OSError as e:
        sys.exit(f"Unable to open training data\n{e}")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"Unable to create output directory: {e}")

    behavior = training_file['behavior']
    window_size = training_file['window_size']
    classifier_type = Classifier.ClassifierType(training_file['classifier_type'])

    if override_classifier is not None:
        classifier_type = override_classifier

    classifier = Classifier()
    if classifier_type in classifier.classifier_choices():
        classifier.set_classifier(classifier_type)
    else:
        print(f"Specified classifier type ({classifier_type.name}) "
              f"is unavailable, using default "
              f"({classifier.classifier_type.name})")

    print("Training classifier for:", behavior)

    training_features = classifier.combine_data(training_file['per_frame'],
                                                training_file['window'])
    classifier.train({
        'training_data': training_features,
        'training_labels': training_file['labels'],
    }, random_seed=training_file['training_seed'])

    # allocate numpy arrays to write to h5 file
    prediction_labels = np.full(
        (pose_est.num_identities, pose_est.num_frames), -1,
        dtype=np.int8)
    prediction_prob = np.zeros_like(prediction_labels, dtype=np.float32)

    print(f"Classifying {input_pose_file}...")
    # print the initial progress bar with 0% complete

    # run prediction for each identity
    for curr_id in pose_est.identities:
        cli_progress_bar(curr_id, len(pose_est.identities),
                         complete_as_percent=False, suffix='identities')
        features = IdentityFeatures(None, curr_id, None, pose_est)
        per_frame_feat = features.get_per_frame()
        window_feat = features.get_window_features(window_size)

        data = Classifier.combine_data(
            per_frame_feat,
            window_feat,
        )

        pred = classifier.predict(data)
        pred_prob = classifier.predict_proba(data)

        # Keep the probability for the predicted class only.
        # The following code uses some
        # numpy magic to use the pred array as column indexes
        # for each row of the pred_prob array we just computed.
        pred_prob = pred_prob[np.arange(len(pred_prob)), pred]

        prediction_labels[curr_id, :] = pred
        prediction_prob[curr_id, :] = pred_prob
    cli_progress_bar(len(pose_est.identities), len(pose_est.identities),
                     complete_as_percent=False, suffix='identities')

    print(f"Writing predictions to {out_dir}")

    behavior_out_dir = out_dir / Project.to_safe_name(behavior)
    behavior_out_dir.mkdir(parents=True, exist_ok=True)
    behavior_out_path = behavior_out_dir / (pose_stem + '.h5')
    with h5py.File(behavior_out_path, 'w') as h5:
        h5.attrs['version'] = Project.PREDICTION_FILE_VERSION
        group = h5.create_group('predictions')
        group.create_dataset('predicted_class', data=prediction_labels)
        group.create_dataset('probabilities', data=prediction_prob)
        group.create_dataset('identity_to_track', data=pose_est.identity_to_track)


def main():

    # find out which classifiers are supported in this environment
    classifier_choices = Classifier().classifier_choices()

    parser = argparse.ArgumentParser()

    classifier_group = parser.add_argument_group(
        "Optionally override the classifier specified in the training file")
    exclusive_group = classifier_group.add_mutually_exclusive_group(required=False)
    for classifer_type, classifier_str in classifier_choices.items():
        exclusive_group.add_argument(
            f"--{classifer_type.name.lower().replace('_', '-')}",
            action='store_const', const=classifer_type,
            dest='classifier', help=f"{classifier_str}"
        )

    parser.add_argument(
        '--training',
        help=f'Training data exported from {APP_NAME}',
        required=True,
    )
    parser.add_argument(
        '--input-pose',
        help='input HDF5 pose file (v2 or v3).',
        required=True,
    )
    parser.add_argument(
        '--out-dir',
        help='directory to store classification output',
        required=True,
    )

    args = parser.parse_args()

    training = Path(args.training)
    out_dir = Path(args.out_dir)
    in_pose_path = Path(args.input_pose)

    classify_pose(training, in_pose_path, out_dir, override_classifier=args.classifier)


if __name__ == "__main__":
    main()
