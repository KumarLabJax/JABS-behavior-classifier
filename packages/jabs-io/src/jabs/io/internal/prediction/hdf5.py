"""HDF5 adapter for BehaviorPrediction using the legacy file layout.

Legacy HDF5 layout::

    / (root)
      attrs: pose_file, pose_hash, version=2
      predictions/ (group)
        external_identity_mapping (dataset, optional)
        <safe_behavior_name>/ (group, one per behavior)
          attrs: classifier_file, classifier_hash, app_version, prediction_date
          predicted_class (dataset, shape: n_identities x n_frames)
          probabilities (dataset, shape: n_identities x n_frames)
          predicted_class_postprocessed (dataset, optional)
          identity_to_track (dataset, optional)

Multiple behaviors coexist in one file. Writes use append mode.
"""

from pathlib import Path

import h5py
import numpy as np

from jabs.core.enums import StorageFormat
from jabs.core.types.prediction import BehaviorPrediction, ClassifierMetadata
from jabs.core.utils import to_safe_name
from jabs.io.base import HDF5Adapter
from jabs.io.registry import register_adapter

_PREDICTION_FILE_VERSION = 2


@register_adapter(StorageFormat.HDF5, BehaviorPrediction, priority=10)
class PredictionHDF5Adapter(HDF5Adapter):
    """HDF5 adapter for BehaviorPrediction using the legacy prediction file layout.

    Overrides ``write`` and ``read`` directly rather than using ``_write_one`` /
    ``_read_one`` because:
    - Writes must use append mode (``"a"``) so multiple behaviors coexist in one file.
    - Reads need an optional ``behavior`` kwarg to select which behavior to load.
    """

    @classmethod
    def can_handle(cls, data_type: type) -> bool:  # noqa: D102
        return data_type is BehaviorPrediction

    # -- write --------------------------------------------------------------

    def _write_one(self, data, group) -> None:
        raise NotImplementedError("Use write() directly for PredictionHDF5Adapter")

    def _read_one(self, group, data_type=None):
        raise NotImplementedError("Use read() directly for PredictionHDF5Adapter")

    def write(  # noqa: D102
        self, data: BehaviorPrediction | list[BehaviorPrediction], path: str | Path, **kwargs
    ) -> None:
        items = data if isinstance(data, list) else [data]
        with h5py.File(path, "a") as h5:
            for pred in items:
                self._write_prediction(pred, h5)

    @staticmethod
    def _write_prediction(pred: BehaviorPrediction, h5: h5py.File) -> None:
        """Write a single BehaviorPrediction into an open HDF5 file."""
        h5.attrs["pose_file"] = pred.pose_file
        h5.attrs["pose_hash"] = pred.pose_hash
        h5.attrs["version"] = _PREDICTION_FILE_VERSION

        prediction_group = h5.require_group("predictions")

        if (
            pred.external_identity_mapping is not None
            and "external_identity_mapping" not in prediction_group
        ):
            prediction_group.create_dataset(
                "external_identity_mapping",
                data=np.array(pred.external_identity_mapping, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        safe_name = to_safe_name(pred.behavior)
        behavior_group = prediction_group.require_group(safe_name)

        behavior_group.attrs["classifier_file"] = pred.classifier.classifier_file
        behavior_group.attrs["classifier_hash"] = pred.classifier.classifier_hash
        behavior_group.attrs["app_version"] = pred.classifier.app_version
        behavior_group.attrs["prediction_date"] = pred.classifier.prediction_date

        _write_dataset(behavior_group, "predicted_class", pred.predicted_class)
        _write_dataset(behavior_group, "probabilities", pred.probabilities)

        if pred.predicted_class_postprocessed is not None:
            _write_dataset(
                behavior_group,
                "predicted_class_postprocessed",
                pred.predicted_class_postprocessed,
            )

        if pred.identity_to_track is not None:
            _write_dataset(behavior_group, "identity_to_track", pred.identity_to_track)
        elif "identity_to_track" in behavior_group:
            del behavior_group["identity_to_track"]

    # -- read ---------------------------------------------------------------

    def read(  # noqa: D102
        self, path: str | Path, data_type: type | None = None, **kwargs
    ) -> BehaviorPrediction | list[BehaviorPrediction]:
        behavior: str | None = kwargs.get("behavior")
        with h5py.File(path, "r") as h5:
            if behavior is not None:
                return self._read_behavior(h5, behavior)
            return self._read_all(h5)

    @staticmethod
    def _read_behavior(h5: h5py.File, behavior: str) -> BehaviorPrediction:
        """Read a single behavior from the legacy HDF5 layout."""
        safe_name = to_safe_name(behavior)
        prediction_group = h5["predictions"]
        behavior_group = prediction_group[safe_name]

        ext_mapping: list[str] | None = None
        if "external_identity_mapping" in prediction_group:
            raw = prediction_group["external_identity_mapping"][()]
            ext_mapping = [v.decode("utf-8") if isinstance(v, bytes) else v for v in raw]

        postprocessed = None
        if "predicted_class_postprocessed" in behavior_group:
            postprocessed = behavior_group["predicted_class_postprocessed"][()]

        identity_to_track = None
        if "identity_to_track" in behavior_group:
            identity_to_track = behavior_group["identity_to_track"][()]

        return BehaviorPrediction(
            behavior=behavior,
            predicted_class=behavior_group["predicted_class"][()],
            probabilities=behavior_group["probabilities"][()],
            classifier=ClassifierMetadata(
                classifier_file=str(behavior_group.attrs["classifier_file"]),
                classifier_hash=str(behavior_group.attrs["classifier_hash"]),
                app_version=str(behavior_group.attrs["app_version"]),
                prediction_date=str(behavior_group.attrs["prediction_date"]),
            ),
            pose_file=str(h5.attrs["pose_file"]),
            pose_hash=str(h5.attrs["pose_hash"]),
            predicted_class_postprocessed=postprocessed,
            identity_to_track=identity_to_track,
            external_identity_mapping=ext_mapping,
        )

    @staticmethod
    def _read_all(h5: h5py.File) -> list[BehaviorPrediction]:
        """Read all behaviors from the legacy HDF5 layout."""
        prediction_group = h5["predictions"]

        ext_mapping: list[str] | None = None
        if "external_identity_mapping" in prediction_group:
            raw = prediction_group["external_identity_mapping"][()]
            ext_mapping = [v.decode("utf-8") if isinstance(v, bytes) else v for v in raw]

        pose_file = str(h5.attrs["pose_file"])
        pose_hash = str(h5.attrs["pose_hash"])

        results: list[BehaviorPrediction] = []
        for key in prediction_group:
            if key == "external_identity_mapping":
                continue
            behavior_group = prediction_group[key]
            if not isinstance(behavior_group, h5py.Group):
                continue

            postprocessed = None
            if "predicted_class_postprocessed" in behavior_group:
                postprocessed = behavior_group["predicted_class_postprocessed"][()]

            identity_to_track = None
            if "identity_to_track" in behavior_group:
                identity_to_track = behavior_group["identity_to_track"][()]

            results.append(
                BehaviorPrediction(
                    behavior=key,
                    predicted_class=behavior_group["predicted_class"][()],
                    probabilities=behavior_group["probabilities"][()],
                    classifier=ClassifierMetadata(
                        classifier_file=str(behavior_group.attrs["classifier_file"]),
                        classifier_hash=str(behavior_group.attrs["classifier_hash"]),
                        app_version=str(behavior_group.attrs["app_version"]),
                        prediction_date=str(behavior_group.attrs["prediction_date"]),
                    ),
                    pose_file=pose_file,
                    pose_hash=pose_hash,
                    predicted_class_postprocessed=postprocessed,
                    identity_to_track=identity_to_track,
                    external_identity_mapping=ext_mapping,
                )
            )
        return results


def _write_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Write or overwrite a dataset in the given group."""
    ds = group.require_dataset(name, shape=data.shape, dtype=data.dtype)
    ds[...] = data
