"""Feature group exposing precomputed V-JEPA embeddings as JABS per-frame features."""

import typing

from jabs.feature_extraction.feature_group_base_class import FeatureGroup
from jabs.pose_estimation import PoseEstimation

from ..feature_base_class import Feature
from .embedding import EmbeddingFeature
from .sidecar import EmbeddingInfo, EmbeddingSidecarError, sidecar_path_for_pose


class EmbeddingFeatureGroup(FeatureGroup):
    """Group backed by a per-video embedding sidecar (see ``sidecar.py``)."""

    _name = "embedding"

    _features: typing.ClassVar[dict[str, type[Feature]]] = {
        EmbeddingFeature.name(): EmbeddingFeature,
    }

    def __init__(
        self, poses: PoseEstimation, pixel_scale: float, window_sizes: tuple[int, ...] = ()
    ) -> None:
        super().__init__(poses, pixel_scale)
        self._embedding_window_sizes = tuple(int(w) for w in window_sizes)

    def _init_feature_mods(self, identity: int) -> dict:
        """Load the sidecar block for ``identity`` and build the embedding feature."""
        sidecar_path = sidecar_path_for_pose(self._poses.pose_file)
        info = EmbeddingInfo(sidecar_path, identity)
        # Fail loud on frame-count disagreement: a silent mismatch would misalign
        # every embedding row against the pose-derived label mask downstream.
        if info.num_frames != self._poses.num_frames:
            raise EmbeddingSidecarError(
                f"sidecar {sidecar_path} has {info.num_frames} frames but pose file "
                f"has {self._poses.num_frames}; regenerate the sidecar for this video"
            )
        return {
            feature: self._features[feature](
                self._poses, self._pixel_scale, info, window_sizes=self._embedding_window_sizes
            )
            for feature in self._enabled_features
        }
