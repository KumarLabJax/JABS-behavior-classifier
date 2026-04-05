"""Tests for the jabs-cli postprocess command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jabs import io
from jabs.core.types.prediction import BehaviorPrediction, ClassifierMetadata
from jabs.scripts.cli.postprocessing import (
    _list_behaviors,
    _load_config_file,
    _stage_template_list,
    generate_config,
    run_apply_postprocessing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLASSIFIER = ClassifierMetadata(
    classifier_file="test.pkl",
    classifier_hash="abc123",
    app_version="0.0.0",
    prediction_date="2024-01-01",
)

_N_FRAMES = 100
_N_IDENTITIES = 2


def _make_prediction(
    behavior: str,
    n_identities: int = _N_IDENTITIES,
    n_frames: int = _N_FRAMES,
    rng: np.random.Generator | None = None,
) -> BehaviorPrediction:
    """Build a minimal BehaviorPrediction with random data."""
    if rng is None:
        rng = np.random.default_rng(seed=0)
    return BehaviorPrediction(
        behavior=behavior,
        predicted_class=rng.integers(0, 2, size=(n_identities, n_frames), dtype=np.int8),
        probabilities=rng.random((n_identities, n_frames)).astype(np.float32),
        classifier=_CLASSIFIER,
        pose_file="pose.h5",
        pose_hash="deadbeef",
        predicted_class_postprocessed=None,
        identity_to_track=None,
        external_identity_mapping=None,
    )


def _write_prediction_file(path: Path, *behaviors: str) -> None:
    """Write one or more BehaviorPrediction objects to an HDF5 file."""
    rng = np.random.default_rng(seed=42)
    for beh in behaviors:
        io.save(_make_prediction(beh, rng=rng), path)


# ---------------------------------------------------------------------------
# _load_config_file
# ---------------------------------------------------------------------------


def test_load_config_file_json_list(tmp_path: Path) -> None:
    """Load a JSON list config."""
    config = [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 5}}]
    cfg_file = tmp_path / "pipeline.json"
    cfg_file.write_text(json.dumps(config))

    result = _load_config_file(cfg_file)

    assert result == config


def test_load_config_file_json_dict(tmp_path: Path) -> None:
    """Load a JSON dict config."""
    config = {
        "grooming": [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 5}}]
    }
    cfg_file = tmp_path / "pipeline.json"
    cfg_file.write_text(json.dumps(config))

    result = _load_config_file(cfg_file)

    assert result == config


def test_load_config_file_invalid_json(tmp_path: Path) -> None:
    """Malformed JSON raises ClickException."""
    import click

    cfg_file = tmp_path / "bad.json"
    cfg_file.write_text("{not valid json")

    with pytest.raises(click.ClickException, match="Invalid JSON"):
        _load_config_file(cfg_file)


def test_load_config_file_unsupported_extension(tmp_path: Path) -> None:
    """An unsupported file extension raises ClickException."""
    import click

    cfg_file = tmp_path / "pipeline.toml"
    cfg_file.write_text("")

    with pytest.raises(click.ClickException, match="Unsupported config file extension"):
        _load_config_file(cfg_file)


# ---------------------------------------------------------------------------
# _list_behaviors
# ---------------------------------------------------------------------------


def test_list_behaviors_single(tmp_path: Path) -> None:
    """Returns the behavior name stored in a single-behavior prediction file."""
    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")

    behaviors = _list_behaviors(pred_file)

    assert behaviors == ["grooming"]


def test_list_behaviors_multiple(tmp_path: Path) -> None:
    """Returns all behavior names when multiple behaviors are present."""
    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming", "rearing")

    behaviors = _list_behaviors(pred_file)

    assert set(behaviors) == {"grooming", "rearing"}


def test_list_behaviors_missing_predictions_group(tmp_path: Path) -> None:
    """A file without a predictions group raises ClickException."""
    import click
    import h5py

    pred_file = tmp_path / "empty.h5"
    with h5py.File(pred_file, "w") as h5:
        h5.attrs["version"] = 2

    with pytest.raises(click.ClickException, match="No 'predictions' group"):
        _list_behaviors(pred_file)


# ---------------------------------------------------------------------------
# run_apply_postprocessing
# ---------------------------------------------------------------------------


def test_run_inplace_writes_postprocessed(tmp_path: Path) -> None:
    """In-place run adds predicted_class_postprocessed to the file."""
    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")

    config = [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 3}}]
    processed = run_apply_postprocessing(pred_file, config, "grooming", pred_file)

    assert processed == ["grooming"]
    result: BehaviorPrediction = io.load(pred_file, BehaviorPrediction, behavior="grooming")
    assert result.predicted_class_postprocessed is not None
    assert result.predicted_class_postprocessed.shape == result.predicted_class.shape


def test_run_output_file_does_not_modify_input(tmp_path: Path) -> None:
    """With --output, the source file is not modified."""
    import h5py

    pred_file = tmp_path / "predictions.h5"
    out_file = tmp_path / "out.h5"
    _write_prediction_file(pred_file, "grooming")

    config = [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 3}}]
    run_apply_postprocessing(pred_file, config, "grooming", out_file)

    # Source file must not have postprocessed dataset
    with h5py.File(pred_file, "r") as h5:
        assert "predicted_class_postprocessed" not in h5["predictions"]["grooming"]

    # Output file must have postprocessed dataset
    with h5py.File(out_file, "r") as h5:
        assert "predicted_class_postprocessed" in h5["predictions"]["grooming"]


def test_run_dict_config_processes_all_behaviors(tmp_path: Path) -> None:
    """Dict config without --behavior processes every behavior in the config."""
    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming", "rearing")

    config = {
        "grooming": [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 3}}],
        "rearing": [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 5}}],
    }
    processed = run_apply_postprocessing(pred_file, config, None, pred_file)

    assert set(processed) == {"grooming", "rearing"}


def test_run_dict_config_behavior_filter(tmp_path: Path) -> None:
    """Dict config with --behavior processes only the specified behavior."""
    import h5py

    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming", "rearing")

    config = {
        "grooming": [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 3}}],
        "rearing": [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 5}}],
    }
    processed = run_apply_postprocessing(pred_file, config, "grooming", pred_file)

    assert processed == ["grooming"]
    with h5py.File(pred_file, "r") as h5:
        assert "predicted_class_postprocessed" in h5["predictions"]["grooming"]
        assert "predicted_class_postprocessed" not in h5["predictions"]["rearing"]


def test_run_list_config_requires_behavior(tmp_path: Path) -> None:
    """List config without --behavior raises ClickException."""
    import click

    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")

    config = [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 3}}]

    with pytest.raises(click.ClickException, match="--behavior"):
        run_apply_postprocessing(pred_file, config, None, pred_file)


def test_run_missing_behavior_in_file_raises(tmp_path: Path) -> None:
    """Requesting a behavior not in the prediction file raises ClickException."""
    import click

    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")

    config = {
        "rearing": [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 3}}]
    }

    with pytest.raises(click.ClickException, match="not found in the prediction file"):
        run_apply_postprocessing(pred_file, config, None, pred_file)


def test_run_invalid_stage_config_raises(tmp_path: Path) -> None:
    """An unrecognised stage name in the config raises ClickException."""
    import click

    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")

    config = [{"stage_name": "NoSuchStage", "parameters": {}}]

    with pytest.raises(click.ClickException, match="Invalid pipeline config"):
        run_apply_postprocessing(pred_file, config, "grooming", pred_file)


# ---------------------------------------------------------------------------
# generate_config / _stage_template_list
# ---------------------------------------------------------------------------


def test_stage_template_list_all_stages_disabled() -> None:
    """All stages in the template have enabled=False."""
    template = _stage_template_list()
    assert all(s["enabled"] is False for s in template)


def test_stage_template_list_has_all_registered_stages() -> None:
    """Template contains an entry for every registered stage."""
    from jabs.behavior.postprocessing.stages import stage_registry

    template = _stage_template_list()
    names = {s["stage_name"] for s in template}
    assert names == set(stage_registry())


def test_generate_config_json_list_format(tmp_path: Path) -> None:
    """Single behavior produces a JSON list config."""
    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")
    out = tmp_path / "pipeline.json"

    generate_config(pred_file, "grooming", out)

    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert all("stage_name" in s and "parameters" in s for s in data)
    assert all(s["enabled"] is False for s in data)


def test_generate_config_json_dict_format(tmp_path: Path) -> None:
    """Multiple behaviors produce a JSON dict config with one key per behavior."""
    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming", "rearing")
    out = tmp_path / "pipeline.json"

    generate_config(pred_file, None, out)

    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert set(data.keys()) == {"grooming", "rearing"}
    for stage_list in data.values():
        assert isinstance(stage_list, list)
        assert all(s["enabled"] is False for s in stage_list)


def test_generate_config_yaml_is_valid(tmp_path: Path) -> None:
    """Generated YAML parses back to the expected structure."""
    pytest.importorskip("yaml")
    import yaml

    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming", "rearing")
    out = tmp_path / "pipeline.yaml"

    generate_config(pred_file, None, out)

    data = yaml.safe_load(out.read_text())
    assert isinstance(data, dict)
    assert set(data.keys()) == {"grooming", "rearing"}


def test_generate_config_behavior_filter(tmp_path: Path) -> None:
    """--behavior restricts the generated config to a single behavior (list format)."""
    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming", "rearing")
    out = tmp_path / "pipeline.json"

    generate_config(pred_file, "grooming", out)

    data = json.loads(out.read_text())
    assert isinstance(data, list)


def test_generate_config_missing_behavior_raises(tmp_path: Path) -> None:
    """Requesting a behavior not in the file raises ClickException."""
    import click

    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")
    out = tmp_path / "pipeline.json"

    with pytest.raises(click.ClickException, match="not found in prediction file"):
        generate_config(pred_file, "rearing", out)


def test_generate_config_unsupported_extension_raises(tmp_path: Path) -> None:
    """An unsupported output extension raises ClickException."""
    import click

    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")
    out = tmp_path / "pipeline.toml"

    with pytest.raises(click.ClickException, match="Unsupported output file extension"):
        generate_config(pred_file, None, out)


def test_run_preserves_raw_predictions(tmp_path: Path) -> None:
    """Postprocessing does not alter the raw predicted_class dataset."""
    pred_file = tmp_path / "predictions.h5"
    _write_prediction_file(pred_file, "grooming")

    original: BehaviorPrediction = io.load(pred_file, BehaviorPrediction, behavior="grooming")

    config = [{"stage_name": "BoutDurationFilterStage", "parameters": {"min_duration": 3}}]
    run_apply_postprocessing(pred_file, config, "grooming", pred_file)

    updated: BehaviorPrediction = io.load(pred_file, BehaviorPrediction, behavior="grooming")
    np.testing.assert_array_equal(updated.predicted_class, original.predicted_class)
    np.testing.assert_array_equal(updated.probabilities, original.probabilities)
