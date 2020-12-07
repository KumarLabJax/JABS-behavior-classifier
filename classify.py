import argparse
import h5py
import numpy as np
from pathlib import Path
import re
import sys
import typing

from src.classifier.skl_classifier import SklClassifier
from src.feature_extraction.features import IdentityFeatures
from src.labeler.project import Project
from src.pose_estimation import open_pose_file


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


def classify_pose(model_proj_dir, input_pose_file, out_dir):
    proj = Project(model_proj_dir, use_cache=False, enable_video_check=False)
    classifier_type = SklClassifier.ClassifierType[proj.metadata['classifier']]
    pose_est = open_pose_file(input_pose_file)
    pose_stem = get_pose_stem(input_pose_file)

    for behavior in proj.metadata['behaviors']:
        curr_label_counts = proj.counts(behavior)
        if SklClassifier.label_threshold_met(curr_label_counts, 1):
            print("Training classifier for:", behavior)

            lbl_feat, _ = proj.get_labeled_features(behavior)
            all_feat = SklClassifier.combine_data(
                lbl_feat['per_frame'],
                lbl_feat['window'])
            classifier = SklClassifier(classifier_type)
            classifier.train({
                'training_data': all_feat,
                'training_labels': lbl_feat['labels'],
            })

            out_dir.mkdir(parents=True, exist_ok=True)

            # allocate numpy arrays to write to h5 file
            prediction_labels = np.full(
                (pose_est.num_identities, pose_est.num_frames), -1,
                dtype=np.int8)
            prediction_prob = np.zeros_like(prediction_labels, dtype=np.float32)

            for curr_id in pose_est.identities:
                print("Predicting for", pose_stem, "ID:", curr_id)

                curr_feat = IdentityFeatures(None, curr_id, None, pose_est)
                per_frame_feat = curr_feat.get_per_frame()
                # TODO hardcoded radius 5 should come from project
                window_feat = curr_feat.get_window_features(5)

                curr_all_feat = SklClassifier.combine_data(
                    per_frame_feat,
                    window_feat,
                )
                pred = classifier.predict(curr_all_feat)
                pred_prob = classifier.predict_proba(curr_all_feat)

                # Keep the probability for the predicted class only.
                # The following code uses some
                # numpy magic to use the pred array as column indexes
                # for each row of the pred_prob array we just computed.
                pred_prob = pred_prob[np.arange(len(pred_prob)), pred]

                prediction_labels[curr_id, :] = pred
                prediction_prob[curr_id, :] = pred_prob

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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--proj-dir',
        help='project directory containing behavior models',
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

    proj_dir = Path(args.proj_dir)
    out_dir = Path(args.out_dir)
    in_pose_path = Path(args.input_pose)

    # if the project dir isn't at rotta project we should abort
    if not (proj_dir / 'rotta').exists():
        print(f'ERROR: "{args.proj_dir}" is not a rotta project directory', file=sys.stderr)
        exit(1)

    proj = Project(proj_dir, enable_video_check=False)

    classify_pose(proj_dir, in_pose_path, out_dir)


if __name__ == "__main__":
    main()
