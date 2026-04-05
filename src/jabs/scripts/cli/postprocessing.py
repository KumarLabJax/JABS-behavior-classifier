"""Apply a postprocessing pipeline to an existing JABS prediction HDF5 file.

Reads predicted classes from the file, runs the configured pipeline stages,
and writes the results back as predicted_class_postprocessed datasets
alongside the original predictions. The raw predictions are never modified.

The --config file is a JSON or YAML file in one of two formats:

\b
  List  — stage configs for a single behavior (requires --behavior).
  Dict  — maps behavior names to stage lists (optionally filtered by --behavior).

Each stage entry requires stage_name and parameters; enabled is optional and
defaults to true. Available stages:

\b
  BoutDurationFilterStage    min_duration (int)           Remove bouts shorter than N frames.
  GapInterpolationStage      max_interpolation_gap (int)  Fill no-prediction gaps up to N frames.
  BoutStitchingStage         max_stitch_gap (int)         Stitch bouts with gaps up to N frames.

Use --list-behaviors to see which behaviors are stored in a prediction file.
For full config format documentation see the JABS user guide.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import h5py
from rich.console import Console

from jabs import io
from jabs.behavior.postprocessing import PostprocessingPipeline
from jabs.behavior.postprocessing.stages import stage_registry
from jabs.core.types.prediction import BehaviorPrediction

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type alias for the per-behavior pipeline config map
_BehaviorConfigMap = dict[str, list[dict[str, Any]]]

# Recommended pipeline order for config generation. Stages present in the
# registry but absent from this list are appended at the end automatically,
# so adding a new stage to the registry will not break generation — but
# developers should add it here with a sensible position and add a default
# value to its KwargHelp entries so the generated template is useful.
_STAGE_ORDER: list[str] = [
    "GapInterpolationStage",
    "BoutStitchingStage",
    "BoutDurationFilterStage",
]


def _stage_template_list() -> list[dict[str, Any]]:
    """Build a template pipeline stage list with all registered stages disabled.

    Stages are ordered per ``_STAGE_ORDER`` (registered but unlisted stages are
    appended at the end). Default parameter values are read from each stage's
    ``KwargHelp.default`` field; parameters whose default is ``None`` are
    omitted from the template.

    Returns:
        List of stage config dicts, each with ``stage_name``, ``enabled: False``,
        and ``parameters``.
    """
    registry = stage_registry()
    ordered = [n for n in _STAGE_ORDER if n in registry]
    remaining = [n for n in registry if n not in _STAGE_ORDER]
    result = []
    for name in ordered + remaining:
        stage_help = registry[name].help()
        parameters = {
            param: kwarg.default
            for param, kwarg in (stage_help.kwargs or {}).items()
            if kwarg.default is not None
        }
        result.append({"stage_name": name, "enabled": False, "parameters": parameters})
    return result


def _serialize_config(
    behaviors: list[str] | None,
    output_path: Path,
) -> str:
    """Serialize a template pipeline config to a JSON or YAML string.

    When *behaviors* contains more than one entry the dict format is used;
    otherwise the list format is used (a single behavior or no prediction file
    context).

    Args:
        behaviors: Behavior names to include as keys in a dict config, or
            ``None`` / a single-element list for list format.
        output_path: Destination path; its suffix determines the output format
            (``.json``, ``.yaml``, or ``.yml``).

    Returns:
        Serialized config as a string.

    Raises:
        click.ClickException: If the output file extension is unsupported.
    """
    registry = stage_registry()
    suffix = output_path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise click.ClickException(
            f"Unsupported output file extension: {suffix!r}. Use .json, .yaml, or .yml."
        )

    use_dict = behaviors is not None and len(behaviors) > 1

    if suffix == ".json":
        config: list | dict = (
            {beh: _stage_template_list() for beh in behaviors}  # type: ignore[union-attr]
            if use_dict
            else _stage_template_list()
        )
        return json.dumps(config, indent=2) + "\n"

    # --- YAML (constructed manually to include comments) ---------------------
    ordered = [n for n in _STAGE_ORDER if n in registry]
    remaining = [n for n in registry if n not in _STAGE_ORDER]
    stage_names = ordered + remaining

    def _stage_block(indent: str) -> list[str]:
        """Render the stage list as YAML lines with the given base indent."""
        lines: list[str] = []
        for name in stage_names:
            stage_help = registry[name].help()
            lines.append(f"{indent}- stage_name: {name}")
            lines.append(f"{indent}  enabled: false")
            lines.append(f"{indent}  # {stage_help.description}")
            lines.append(f"{indent}  parameters:")
            for param, kwarg_help in (stage_help.kwargs or {}).items():
                if kwarg_help.default is None:
                    continue
                comment = f"  # {kwarg_help.description}"
                lines.append(f"{indent}    {param}: {kwarg_help.default}{comment}")
        return lines

    header = [
        "# Postprocessing pipeline config for jabs-cli postprocess.",
        "# Set 'enabled: true' to activate a stage. Stages run in the order listed.",
        "#",
        "# Recommended order: GapInterpolationStage / BoutStitchingStage before",
        "#                    BoutDurationFilterStage.",
        "",
    ]

    body: list[str] = []
    if use_dict:
        for behavior in behaviors:  # type: ignore[union-attr]
            body.append(f"{behavior}:")
            body.extend(_stage_block("  "))
            body.append("")
    else:
        body.extend(_stage_block(""))

    return "\n".join(header + body) + "\n"


def generate_config(
    prediction_file: Path,
    behavior: str | None,
    output_path: Path,
) -> None:
    """Generate a template pipeline config file from a prediction file.

    Writes a config pre-populated with all registered stages (disabled by
    default) and the behavior names present in *prediction_file*.  If
    *behavior* is provided only that behavior is included; otherwise all
    behaviors found in the file are included.

    Args:
        prediction_file: Path to the JABS prediction HDF5 file.
        behavior: If provided, generate a single-behavior (list) config for
            this behavior only.
        output_path: Destination file path (.json, .yaml, or .yml).

    Raises:
        click.ClickException: If the prediction file cannot be read, a
            requested behavior is not found, or the output extension is
            unsupported.
    """
    available = _list_behaviors(prediction_file)

    if behavior is not None:
        if behavior not in available:
            raise click.ClickException(
                f"Behavior '{behavior}' not found in prediction file. "
                f"Available behaviors: {', '.join(repr(b) for b in available)}"
            )
        behaviors: list[str] | None = [behavior]
    else:
        behaviors = available if len(available) > 1 else (available or None)

    content = _serialize_config(behaviors, output_path)
    try:
        output_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"Cannot write config file: {exc}") from exc


def _load_config_file(config_path: Path) -> list[dict[str, Any]] | _BehaviorConfigMap:
    """Load a pipeline config from a JSON or YAML file.

    Args:
        config_path: Path to the JSON or YAML config file.

    Returns:
        A list of stage configs (single-behavior format) or a dict mapping
        behavior names to lists of stage configs (multi-behavior format).

    Raises:
        click.ClickException: If the file extension is unsupported, PyYAML is
            not installed when a YAML file is requested, or the file cannot be
            parsed.
    """
    suffix = config_path.suffix.lower()
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"Cannot read config file: {exc}") from exc

    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"Invalid JSON in config file: {exc}") from exc
    elif suffix in {".yaml", ".yml"}:
        if not _YAML_AVAILABLE:
            raise click.ClickException(
                "PyYAML is required for YAML config files but is not installed. "
                "Install the yaml extra: pip install 'jabs-behavior-classifier[yaml]'"
            )
        try:
            return _yaml.safe_load(text)
        except Exception as exc:
            raise click.ClickException(f"Invalid YAML in config file: {exc}") from exc
    else:
        raise click.ClickException(
            f"Unsupported config file extension: {suffix!r}. Use .json, .yaml, or .yml."
        )


def _list_behaviors(prediction_path: Path) -> list[str]:
    """Return the behavior names (safe names) present in a prediction HDF5 file.

    Args:
        prediction_path: Path to the JABS prediction HDF5 file.

    Returns:
        List of behavior name strings (safe-name HDF5 group keys).

    Raises:
        click.ClickException: If the file cannot be opened or has no predictions
            group.
    """
    try:
        with h5py.File(prediction_path, "r") as h5:
            if "predictions" not in h5:
                raise click.ClickException(
                    f"No 'predictions' group found in {prediction_path}. "
                    "Is this a valid JABS prediction file?"
                )
            prediction_group = h5["predictions"]
            return [
                key
                for key in prediction_group
                if key != "external_identity_mapping"
                and isinstance(prediction_group[key], h5py.Group)
            ]
    except OSError as exc:
        raise click.ClickException(f"Cannot open prediction file: {exc}") from exc


def run_apply_postprocessing(
    prediction_file: Path,
    config: list[dict[str, Any]] | _BehaviorConfigMap,
    behavior_name: str | None,
    output_path: Path,
    on_stage: Callable[[str, str], None] | None = None,
) -> list[str]:
    """Apply a postprocessing pipeline to predictions in a JABS HDF5 file.

    Reads the raw predicted classes, applies the configured pipeline stages,
    and writes ``predicted_class_postprocessed`` back to *output_path*.

    Args:
        prediction_file: Path to the source JABS prediction HDF5 file.
        config: Pipeline config — either a list of stage dicts (single-behavior)
            or a dict mapping behavior names to lists of stage dicts.
        behavior_name: If provided, restrict processing to this behavior.
            Required when *config* is a list.
        output_path: Destination path for the updated prediction file.
            May be equal to *prediction_file* for in-place updates.
        on_stage: Optional callback invoked just before each stage runs.
            Receives the behavior name and stage class name as arguments.

    Returns:
        List of behavior names that were successfully processed.

    Raises:
        click.ClickException: On invalid config, missing behaviors, or I/O
            errors.
    """
    # --- Resolve per-behavior pipeline config map ----------------------------
    if isinstance(config, list):
        if behavior_name is None:
            raise click.ClickException(
                "The config file contains a list of stages (single-behavior format). "
                "Use --behavior to specify which behavior to apply the pipeline to."
            )
        behavior_configs: _BehaviorConfigMap = {behavior_name: config}
    elif isinstance(config, dict):
        if behavior_name is not None:
            if behavior_name not in config:
                available = ", ".join(f"'{b}'" for b in config)
                raise click.ClickException(
                    f"Behavior '{behavior_name}' not found in config file. "
                    f"Available behaviors in config: {available}"
                )
            behavior_configs = {behavior_name: config[behavior_name]}
        else:
            behavior_configs = config
    else:
        raise click.ClickException(
            "Config file must contain a JSON/YAML list or object at the top level."
        )

    # --- Validate behaviors against the prediction file ----------------------
    available_in_file = _list_behaviors(prediction_file)
    missing = [b for b in behavior_configs if b not in available_in_file]
    if missing:
        raise click.ClickException(
            f"The following behavior(s) were not found in the prediction file: "
            f"{', '.join(repr(b) for b in missing)}. "
            f"Available behaviors: {', '.join(repr(b) for b in available_in_file)}"
        )

    # --- Prepare output file -------------------------------------------------
    if output_path != prediction_file:
        try:
            shutil.copy2(prediction_file, output_path)
        except OSError as exc:
            raise click.ClickException(f"Cannot write output file: {exc}") from exc
        logger.info("Copied '%s' to '%s'", prediction_file, output_path)

    # --- Process each behavior -----------------------------------------------
    processed: list[str] = []
    for behavior, stage_config in behavior_configs.items():
        try:
            pipeline = PostprocessingPipeline(stage_config)
        except ValueError as exc:
            raise click.ClickException(
                f"Invalid pipeline config for behavior '{behavior}': {exc}"
            ) from exc

        try:
            pred: BehaviorPrediction = io.load(
                prediction_file, BehaviorPrediction, behavior=behavior
            )
        except Exception as exc:
            raise click.ClickException(
                f"Failed to read predictions for behavior '{behavior}' "
                f"from '{prediction_file}': {exc}"
            ) from exc

        n_identities, _n_frames = pred.predicted_class.shape
        postprocessed = pred.predicted_class.copy()
        for stage in pipeline.stages:
            if on_stage is not None:
                on_stage(behavior, type(stage).__name__)
            for i in range(n_identities):
                postprocessed[i] = stage.apply(postprocessed[i], pred.probabilities[i])

        updated_pred = BehaviorPrediction(
            behavior=pred.behavior,
            predicted_class=pred.predicted_class,
            probabilities=pred.probabilities,
            classifier=pred.classifier,
            pose_file=pred.pose_file,
            pose_hash=pred.pose_hash,
            predicted_class_postprocessed=postprocessed,
            identity_to_track=pred.identity_to_track,
            external_identity_mapping=pred.external_identity_mapping,
        )
        try:
            io.save(updated_pred, output_path)
        except Exception as exc:
            raise click.ClickException(
                f"Failed to write postprocessed predictions for behavior '{behavior}' "
                f"to '{output_path}': {exc}"
            ) from exc
        logger.info("Wrote postprocessed predictions for behavior '%s'", behavior)
        processed.append(behavior)

    return processed


@click.command(
    name="postprocess",
    context_settings={"max_content_width": 120},
    help=__doc__,
)
@click.argument(
    "prediction_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--config",
    "config_file",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to a JSON or YAML file describing the postprocessing pipeline. "
        "May be a list of stages (single-behavior) or a dict mapping behavior "
        "names to lists of stages (multi-behavior). Required unless "
        "--list-behaviors is used."
    ),
)
@click.option(
    "--behavior",
    default=None,
    type=str,
    help=(
        "Restrict processing to a single behavior. Required when the config "
        "file contains a list of stages. When the config is a dict, filters "
        "to only the specified behavior."
    ),
)
@click.option(
    "--output",
    "output_file",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help=("Path for the output prediction file. If omitted, the input file is updated in place."),
)
@click.option(
    "--list-behaviors",
    is_flag=True,
    default=False,
    help="List the behaviors present in the prediction file and exit.",
)
@click.option(
    "--generate-config",
    "generate_config_file",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help=(
        "Write a template pipeline config to this path and exit. "
        "Format is determined by the file extension (.json, .yaml, or .yml). "
        "The template includes all available stages (disabled by default) "
        "pre-populated with the behavior names from the prediction file. "
        "Cannot be combined with --config."
    ),
)
@click.pass_context
def apply_postprocessing_command(
    ctx: click.Context,
    prediction_file: Path,
    config_file: Path | None,
    behavior: str | None,
    output_file: Path | None,
    list_behaviors: bool,
    generate_config_file: Path | None,
) -> None:
    """Apply a postprocessing pipeline to a JABS prediction HDF5 file."""
    if list_behaviors:
        behaviors = _list_behaviors(prediction_file)
        if behaviors:
            click.echo("Behaviors in prediction file:")
            for b in behaviors:
                click.echo(f"  {b}")
        else:
            click.echo("No behaviors found in prediction file.")
        return

    if generate_config_file is not None:
        if config_file is not None:
            raise click.UsageError("--generate-config and --config cannot be used together.")
        generate_config(prediction_file, behavior, generate_config_file)
        click.echo(f"Wrote template config to {generate_config_file}")
        return

    if config_file is None:
        raise click.UsageError(
            "--config is required unless --list-behaviors or --generate-config is used."
        )

    config = _load_config_file(config_file)
    output_path = output_file if output_file is not None else prediction_file
    in_place = output_path == prediction_file

    if ctx.obj["VERBOSE"]:
        click.echo(f"Prediction file: {prediction_file}")
        click.echo(f"Config file:     {config_file}")
        click.echo(f"Output:          {output_path} {'(in place)' if in_place else ''}")
        if behavior:
            click.echo(f"Behavior filter: {behavior}")

    console = Console()

    # show status in the console -- unless the input file is very large this will only appear for a brief moment,
    # possibly not even noticeable to the user, but for large input files it could provide useful feedback
    # for long-running stages and reassures the user that something is happening
    with console.status("Starting...", spinner="dots") as status:

        def on_stage(behavior_name: str, stage_name: str) -> None:
            status.update(f"[bold]{behavior_name}[/bold] — {stage_name}")

        processed = run_apply_postprocessing(
            prediction_file, config, behavior, output_path, on_stage=on_stage
        )

    for processed_behavior in processed:
        click.echo(f"Postprocessed '{processed_behavior}' \u2192 {output_path}")
