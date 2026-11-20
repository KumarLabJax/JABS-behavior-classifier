"""Golden per-frame values of the segmentation-dependent features.

The goldens pin the segmentation features (moments, Hu moments, shape
descriptors) and the social features (which read segmentation contours for
contact and huddle features) computed on ``tests/data/sample_pose_est_v6.h5``.
They were recorded from the features reading whole ``-1``-padded segmentation
arrays (``get_segmentation_data``), before the features moved to segmentation
windows, so the window-based features are checked against the old values.

The goldens are kept small: values are recorded for every
``FRAME_STRIDE``-th frame only, as float32. Features are still computed over
every frame, so frame-to-frame features (e.g. centroid speed) are covered.

Recording goldens:
    Goldens are committed, never recorded by a test run. To (re)record them
    after an intended behavior change, run from the repo root::

        uv run python tests/feature_extraction/segmentation_features/feature_goldens.py --record

    and review the change before committing it.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from jabs.core.pose_protocol import PoseLike
from jabs.feature_extraction.segmentation_features import SegmentationFeatureGroup
from jabs.feature_extraction.social_features import SocialFeatureGroup

logger = logging.getLogger(__name__)

SAMPLE_PATH = Path(__file__).parent.parent.parent / "data" / "sample_pose_est_v6.h5"
GOLDEN_PATH = Path(__file__).parent / "goldens" / "sample_pose_est_v6_features.npz"

# frames between recorded frames
FRAME_STRIDE = 20
# pixel scale the features are computed with; not 1.0 so pixel scaling is covered
PIXEL_SCALE = 0.05

_GROUPS = (SegmentationFeatureGroup, SocialFeatureGroup)


def compute_features(pose: PoseLike) -> dict[str, npt.NDArray[np.float32]]:
    """Compute the per-frame segmentation and social features of every identity.

    Args:
        pose: The pose to compute features for.

    Returns:
        Mapping of ``"<identity>/<group>/<module>/<feature>"`` to the feature's
        values on every ``FRAME_STRIDE``-th frame, as float32.
    """
    values: dict[str, npt.NDArray[np.float32]] = {}
    for group_class in _GROUPS:
        group = group_class(pose, PIXEL_SCALE)
        for identity in pose.identities:
            for module, features in group.per_frame(identity).items():
                for feature, array in features.items():
                    key = f"{identity}/{group_class.name()}/{module}/{feature}"
                    values[key] = np.asarray(array, dtype=np.float32)[::FRAME_STRIDE]
    return values


def load_golden(path: Path = GOLDEN_PATH) -> dict[str, npt.NDArray[Any]]:
    """Load the recorded feature values.

    Args:
        path: The golden ``.npz`` file.

    Returns:
        Mapping of feature key to recorded values.
    """
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def record_golden(path: Path = GOLDEN_PATH) -> Path:
    """Record the goldens from the sample v6 file.

    Args:
        path: Destination ``.npz`` path; its directory is created if needed.

    Returns:
        The written path.
    """
    # imported here: recording opens the file, checking is given a pose
    from jabs.pose_estimation import open_pose_file

    with open_pose_file(SAMPLE_PATH) as pose:
        values = compute_features(pose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    logger.info("Recorded %s (%d bytes)", path, path.stat().st_size)
    return path


def main(argv: list[str] | None = None) -> None:
    """Command line entry point; ``--record`` rewrites the goldens.

    Args:
        argv: Command line arguments, defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true", help="record the feature goldens")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.record:
        parser.error("nothing to do; pass --record to rewrite the goldens")
    record_golden()


if __name__ == "__main__":
    main()
