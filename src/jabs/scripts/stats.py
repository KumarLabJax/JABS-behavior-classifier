import argparse
import itertools

import numpy as np
from tabulate import tabulate

from jabs.classifier import Classifier
from jabs.project import load_training_data
from jabs.scripts.classify import train
from jabs.types import ProjectDistanceUnit


def main():
    """jabs-stats"""
    parser = argparse.ArgumentParser(
        prog="stats",
        description="print accuracy statistics for the given classifier",
    )

    parser.add_argument(
        "-k",
        help="the parameter controlling the maximum number of iterations."
        " Default is to iterate over all leave-one-out possibilities.",
        type=int,
    )

    parser.add_argument(
        "training",
        help="training data HDF5 file",
    )

    args = parser.parse_args()

    classifier = train(args.training)

    features, group_mapping = load_training_data(args.training)
    data_generator = Classifier.leave_one_group_out(
        features["per_frame"],
        features["window"],
        features["labels"],
        features["groups"],
    )

    table_rows = []
    accuracies = []
    fbeta_behavior = []
    fbeta_notbehavior = []

    iter_count = 0
    for i, data in enumerate(itertools.islice(data_generator, args.k)):
        iter_count += 1

        test_info = group_mapping[data["test_group"]]

        # train classifier, and then use it to classify our test data
        classifier.train(data)
        predictions = classifier.predict(data["test_data"])

        # calculate some performance metrics using the classifications of
        # the test data
        accuracy = classifier.accuracy_score(data["test_labels"], predictions)
        pr = classifier.precision_recall_score(data["test_labels"], predictions)
        confusion = classifier.confusion_matrix(data["test_labels"], predictions)

        table_rows.append(
            [
                accuracy,
                pr[0][0],
                pr[0][1],
                pr[1][0],
                pr[1][1],
                pr[2][0],
                pr[2][1],
                f"{test_info['video']} [{test_info['identity']}]",
            ]
        )
        accuracies.append(accuracy)
        fbeta_behavior.append(pr[2][1])
        fbeta_notbehavior.append(pr[2][0])

        # print performance metrics and feature importance to console
        print("-" * 70)
        print(f"training iteration {i}")
        print("TEST DATA:")
        print(f"\tVideo: {test_info['video']}")
        print(f"\tIdentity: {test_info['identity']}")
        print(f"ACCURACY: {accuracy * 100:.2f}%")
        print("PRECISION RECALL:")
        print(f"              {'not behavior':12}  behavior")
        print(f"  precision   {pr[0][0]:<12.8}  {pr[0][1]:<.8}")
        print(f"  recall      {pr[1][0]:<12.8}  {pr[1][1]:<.8}")
        print(f"  F1 score    {pr[2][0]:<12.8}  {pr[2][1]:<.8}")
        print(f"  support     {pr[3][0]:<12}  {pr[3][1]}")
        print("CONFUSION MATRIX:")
        print(f"{confusion}")
        print("-" * 70)

        print("Top 10 features by importance:")
        classifier.print_feature_importance(data["feature_names"], 10)

    if iter_count >= 1:
        print("\n" + "=" * 70)
        print("SUMMARY\n")
        print(
            tabulate(
                table_rows,
                showindex="always",
                headers=[
                    "accuracy",
                    "precision\n(not behavior)",
                    "precision\n(behavior)",
                    "recall\n(not behavior)",
                    "recall\n(behavior)",
                    "F1 score\n(not behavior)",
                    "F1 score\n(behavior)",
                    "test - leave one out:\n(video [identity])",
                ],
            )
        )

        print(f"\nmean accuracy: {np.mean(accuracies):.5}")
        print(f"mean F1 score (behavior): {np.mean(fbeta_behavior):.5}")
        print(f"mean F1 score (not behavior): {np.mean(fbeta_notbehavior):.5}")
        print(f"\nClassifier: {classifier.classifier_name}")
        print(f"Behavior: {features['behavior']}")
        unit = (
            "cm" if classifier.project_settings["cm_units"] == ProjectDistanceUnit.CM else "pixel"
        )
        print(f"Feature Distance Unit: {unit}")
        print("-" * 70)
    else:
        print("No results calculated")


if __name__ == "__main__":
    main()
