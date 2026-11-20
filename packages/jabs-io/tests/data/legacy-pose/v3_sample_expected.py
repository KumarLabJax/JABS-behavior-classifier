"""Record what today's ``PoseEstimationV3`` reads from the two public v3 samples.

This script needs the legacy readers in ``jabs.pose_estimation`` as they were
before Task 16 of the "JABS adopts the pose file" plan removed them; it is a
record of how ``v3-sample-expected.npz`` was made, not runnable tooling after
that. The jabs-io tests only load the ``.npz``; they never import
``jabs.pose_estimation``.

It was run from the repository root::

    uv run python packages/jabs-io/tests/data/legacy-pose/v3_sample_expected.py

For each sample, keyed ``<file stem>/<name>``, the archive holds:

* ``identity_to_track``: ``(I, F)`` int32, the legacy track id each identity
  holds in each frame, -1 where it holds none, over every frame.
* ``point_mask``: ``(I, F, 12)`` bool, confidence above 0.3, over every frame.
* ``identity_mask``: ``(I, F)`` bool, at least one body keypoint valid.
* ``frames``: the frame subset ``points`` is recorded for, every
  ``FRAME_STRIDE``-th frame.
* ``points``: ``(I, len(frames), 12, 2)`` float32 ``(x, y)``; the samples
  store ``uint16``, so float32 holds them exactly.

Today's reader mishandles a track id that ends and comes back (it keeps the
first run's identity and stacks the runs' frames contiguously). Neither sample
has one, which this script asserts, so every recorded frame is comparable.
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from jabs.pose_estimation.pose_est_v3 import PoseEstimationV3

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
SAMPLES = HERE.parents[4] / "tests" / "data"
STEMS = ("sample_pose_est_v3", "identity_with_no_data_pose_est_v3")
OUTPUT = HERE / "v3-sample-expected.npz"
FRAME_STRIDE = 10


def _assert_no_reappearing_track(path: Path) -> None:
    """Fail when any track id is absent between two frames that hold it."""
    with h5py.File(path, "r") as h5:
        count = h5["poseest/instance_count"][()]
        track = h5["poseest/instance_track_id"][()]
    frames: dict[int, list[int]] = {}
    for frame in range(count.size):
        for track_id in track[frame, : count[frame]].tolist():
            frames.setdefault(track_id, []).append(frame)
    for track_id, held in frames.items():
        assert np.all(np.diff(held) == 1), f"{path.name}: track {track_id} reappears"


def _record(stem: str) -> dict[str, npt.NDArray[np.generic]]:
    """Return one sample's arrays, keyed ``<stem>/<name>``."""
    path = SAMPLES / f"{stem}.h5"
    _assert_no_reappearing_track(path)
    pose = PoseEstimationV3(path)
    identities = range(pose.num_identities)
    frames = np.arange(0, pose.num_frames, FRAME_STRIDE)
    points = np.stack([pose.get_identity_poses(i)[0][frames] for i in identities])
    arrays = {
        "identity_to_track": pose.identity_to_track.astype(np.int32),
        "point_mask": np.stack([pose.get_identity_point_mask(i) for i in identities]).astype(bool),
        "identity_mask": np.stack([pose.identity_mask(i) for i in identities]).astype(bool),
        "frames": frames,
        "points": points.astype(np.float32),
    }
    logger.info("%s: %d identities, %d frames", stem, pose.num_identities, pose.num_frames)
    return {f"{stem}/{name}": value for name, value in arrays.items()}


def main() -> None:
    """Write ``v3-sample-expected.npz`` beside this script."""
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    for stem in STEMS:
        arrays.update(_record(stem))
    np.savez_compressed(OUTPUT, **arrays)
    logger.info("wrote %s, %d bytes", OUTPUT, OUTPUT.stat().st_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
