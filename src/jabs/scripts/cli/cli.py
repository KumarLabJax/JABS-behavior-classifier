"""The JABS CLI

This module provides a command-line interface to interact with a JABS project.
Initially, it was developed as a collect-all of smaller utilities that didn't
warrant their own scripts, but it might eventually evolve into a more
comprehensive tool for managing JABS projects (possibly integrating
functionality from existing JABS scripts such as `jabs-init` or `jabs-merge`).
"""

import json
from pathlib import Path

import click
from rich.console import Console

from jabs.classifier import Classifier, MlflowLoggingError, mlflow_available, parse_kv_tags
from jabs.core.enums import ClassifierMode, ClassifierType, CrossValidationGroupingStrategy
from jabs.project import (
    Project,
    export_training_data,
    export_training_data_multiclass,
    get_videos_to_prune,
)

from .compute_features import compute_features_command
from .convert_parquet import convert_parquet_command
from .convert_to_nwb import run_conversion
from .cross_validation import run_cross_validation
from .postprocessing import apply_postprocessing_command
from .sample_frames import sample_frames_command
from .sample_pose_intervals import sample_pose_intervals_command
from .update_labels import update_labels_command
from .update_pose import update_pose_command

# find out which classifiers are supported in this environment
CLASSIFIER_CHOICES: list[ClassifierType] = Classifier().classifier_choices()
DEFAULT_CLASSIFIER: str = (
    ClassifierType.XGBOOST.value.lower()
    if ClassifierType.XGBOOST in CLASSIFIER_CHOICES
    else ClassifierType.RANDOM_FOREST.value.lower()
)


@click.group(context_settings={"max_content_width": 120})
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
@click.pass_context
def cli(ctx: click.Context, verbose):
    """JABS CLI."""
    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose


cli.add_command(apply_postprocessing_command)
cli.add_command(compute_features_command)
cli.add_command(convert_parquet_command)
cli.add_command(update_pose_command)
cli.add_command(update_labels_command)
cli.add_command(sample_pose_intervals_command)
cli.add_command(sample_frames_command)


@cli.command(name="export-training")
@click.argument(
    "directory",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--behavior",
    required=False,
    default=None,
    type=str,
    help="Behavior to export. Required for binary-mode projects; ignored for multi-class projects.",
)
@click.option(
    "--classifier",
    "classifier",
    default="xgboost"
    if ClassifierType.XGBOOST in CLASSIFIER_CHOICES
    else ClassifierType.RANDOM_FOREST.name.lower(),
    type=click.Choice([c.name for c in CLASSIFIER_CHOICES], case_sensitive=False),
    help="Default classifier set in the training file. Default is 'xgboost'.",
)
@click.option(
    "--outfile",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=False,
    help=(
        "Optional path to write the exported training data file. "
        "Default is <project_dir>/<behavior>_training_<YYYYMMDD_HHMMSS>.h5 (binary) "
        "or <project_dir>/multiclass_training_<YYYYMMDD_HHMMSS>.h5 (multi-class)."
    ),
)
@click.pass_context
def export_training(
    ctx: click.Context,
    directory: Path,
    behavior: str | None,
    classifier: str,
    outfile: Path | None,
):
    """Export training data for a JABS project directory.

    For binary-mode projects, --behavior is required. For multi-class projects, all
    behaviors are exported together and --behavior is not used.
    """
    classifier_type = ClassifierType[classifier.upper()]
    jabs_project = Project(directory, enable_session_tracker=False)
    classifier_mode = jabs_project.settings_manager.classifier_mode

    if classifier_mode == ClassifierMode.MULTICLASS:
        if behavior is not None:
            click.echo("Note: --behavior is ignored for multi-class projects.")
        if ctx.obj["VERBOSE"]:
            click.echo("Exporting multi-class training data with the following parameters:")
            click.echo(f"\tClassifier type: {classifier}")
            click.echo(f"\tJABS project directory: {directory}")
        console = Console()
        status_text = f"Exporting multi-class training data (classifier={classifier_type.name})"
        with console.status(status_text, spinner="dots"):
            outfile = export_training_data_multiclass(
                jabs_project,
                jabs_project.feature_manager.min_pose_version,
                classifier_type,
                out_file=outfile,
            )
    else:
        if behavior is None:
            raise click.ClickException("--behavior is required for binary-mode projects.")
        if behavior not in jabs_project.settings["behavior"]:
            raise click.ClickException(f"Behavior '{behavior}' not found in project.")
        if ctx.obj["VERBOSE"]:
            click.echo("Exporting training data with the following parameters:")
            click.echo(f"\tBehavior: {behavior}")
            click.echo(f"\tClassifier type: {classifier}")
            click.echo(f"\tJABS project directory: {directory}")
        console = Console()
        status_text = (
            f"Exporting training data (behavior={behavior}, classifier={classifier_type.name})"
        )
        with console.status(status_text, spinner="dots"):
            outfile = export_training_data(
                jabs_project,
                behavior,
                jabs_project.feature_manager.min_pose_version,
                classifier_type,
                out_file=outfile,
            )

    click.echo(f"Exported training data to {outfile}")


@cli.command(name="rename-behavior")
@click.argument(
    "directory",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.argument(
    "old-name",
    type=str,
)
@click.argument(
    "new-name",
    type=str,
)
@click.pass_context
def rename_behavior(ctx: click.Context, directory: Path, old_name: str, new_name: str) -> None:
    """Rename a behavior in a JABS project."""
    # Args:
    #    ctx: Click context.
    #    directory (Path): Path to the JABS project directory.
    #    old_name (str): Current name of the behavior to rename.
    #    new_name (str): New name for the behavior.
    #
    # Raises:
    #    click.ClickException: If the old behavior name does not exist or
    #        the new behavior name already exists.

    if ctx.obj["VERBOSE"]:
        click.echo("Renaming behavior with the following parameters:")
        click.echo(f"\tOld behavior name: {old_name}")
        click.echo(f"\tNew behavior name: {new_name}")
        click.echo(f"\tJABS project directory: {directory}")

    if not Project.is_valid_project_directory(directory):
        raise click.ClickException(f"Invalid JABS project directory: {directory}")

    jabs_project = Project(directory, enable_session_tracker=False)

    # validate that the old behavior exists in the project
    if old_name not in jabs_project.settings["behavior"]:
        raise click.ClickException(f"Behavior '{old_name}' not found in project.")

    # validate that the new behavior does not already exist in the project
    if new_name in jabs_project.settings["behavior"]:
        raise click.ClickException(f"Behavior '{new_name}' already exists in project.")

    console = Console()
    status_text = f"Renaming behavior '{old_name}' to '{new_name}'"
    with console.status(status_text, spinner="dots"):
        jabs_project.rename_behavior(old_name, new_name)


@cli.command(name="prune")
@click.argument(
    "directory",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--behavior",
    type=str,
    default=None,
    help="Filter by behavior name. If provided, only videos labeled for this behavior will be retained; otherwise, all videos with any labeled behavior are kept.",
)
@click.pass_context
def prune(ctx: click.Context, directory: Path, behavior: str | None):
    """Prune unused videos from a JABS project directory."""
    if not Project.is_valid_project_directory(directory):
        raise click.ClickException(f"Invalid JABS project directory: {directory}")

    project = Project(directory)
    videos_to_prune = get_videos_to_prune(project, behavior)

    if not videos_to_prune:
        click.echo("No videos to prune.")
        return

    click.echo(
        f"Found {len(videos_to_prune)} videos to prune out of {len(project.video_manager.videos)} total videos."
    )
    click.echo("The following videos will be removed:")
    for video_path in videos_to_prune:
        click.echo(f" - {video_path.video_path.name}")

    confirm = click.confirm("Do you want to proceed with pruning these videos?", default=False)

    if not confirm:
        click.echo("Pruning cancelled.")
        return

    for video_paths in videos_to_prune:
        # get related files that also need to be cleaned up
        derived_files = project.get_derived_file_paths(video_paths.video_path.name)

        # Remove files, ignore file not found errors
        try:
            video_paths.video_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            click.echo(f"Warning: failed to delete video or pose file: {e}")

        try:
            video_paths.pose_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            click.echo(f"Warning: failed to delete video or pose file: {e}")

        # remove parquet files too, these aren't used by JABS but may be present in
        # envision derived JABS projects
        parquet_path = video_paths.video_path.with_suffix(".parquet")
        try:
            parquet_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            click.echo(f"Warning: failed to delete parquet file: {e}")

        for file in derived_files:
            try:
                file.unlink()
            except FileNotFoundError:
                continue
            except Exception as e:
                click.echo(f"Warning: failed to delete derived file {file}: {e}")

        # remove from the project.json file
        project.settings_manager.remove_video_from_project_file(
            video_paths.video_path.name, sync=False
        )
    project.settings_manager.save_project_file()


@cli.command(name="cross-validation")
@click.argument(
    "directory",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "-k",
    type=int,
    default=0,
    show_default=True,
    help="Number of cross-validation splits (0 for all splits).",
)
@click.option(
    "--behavior",
    type=str,
    required=True,
    help="Behavior to perform cross-validation on (required). Can be quoted if it contains spaces. "
    "Must match an existing behavior in the project.",
)
@click.option(
    "--grouping-strategy",
    type=click.Choice(["video", "individual", "filename"], case_sensitive=False),
    default=None,
    help=(
        "Cross validation grouping strategy. If not provided, use the project setting. "
        "The 'filename' strategy groups videos by a regular expression applied to their "
        "filenames and requires --grouping-pattern (or a pattern saved in the project)."
    ),
)
@click.option(
    "--grouping-pattern",
    "grouping_pattern",
    type=str,
    default=None,
    help=(
        "Regular expression used to extract a grouping key from each video filename. "
        "Only used with '--grouping-strategy filename'. If not provided, the pattern saved "
        "in the project settings is used."
    ),
)
@click.option(
    "--classifier",
    "classifier",
    default=DEFAULT_CLASSIFIER,
    type=click.Choice([c.name for c in CLASSIFIER_CHOICES], case_sensitive=False),
    help=f"Default classifier set in the training file. Default is '{DEFAULT_CLASSIFIER}'.",
)
@click.option(
    "--report-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Optional path to write the cross-validation report. "
    "Report format will be determined by extension (.md for Markdown, .json for JSON). "
    "If not provided, a default filename will be used.",
)
@click.option(
    "--mlflow",
    "mlflow_env",
    is_flag=False,
    flag_value="",
    default=None,
    metavar="ENV_FILE",
    help="Enable opt-in MLflow logging of the cross-validation results (aggregate "
    "metrics, params, and the training report artifact). Optionally takes a path to a "
    ".env file holding MLFLOW_* settings (tracking URI, experiment, auth, TLS); with no "
    "path, those are read from the ambient environment. Absent leaves the command's "
    "behavior unchanged. Requires the optional 'mlflow' extra "
    "(pip install 'jabs-behavior-classifier[mlflow]').",
)
@click.option(
    "--mlflow-experiment",
    "mlflow_experiment",
    type=str,
    default=None,
    metavar="NAME",
    help="With --mlflow, the MLflow experiment to log the run under. If not provided, "
    "defaults to the MLFLOW_EXPERIMENT_NAME environment variable, else 'jabs-<behavior>' "
    "(one experiment per behavior). No-op without --mlflow.",
)
@click.option(
    "--mlflow-tag",
    "mlflow_tags",
    multiple=True,
    metavar="KEY=VALUE",
    help="With --mlflow, add a free-form run tag (repeatable), e.g. purpose=baseline. "
    "Merges over the auto-derived tags. No-op without --mlflow.",
)
@click.option(
    "--mlflow-no-report",
    "mlflow_no_report",
    is_flag=True,
    help="With --mlflow, skip uploading the training report artifact (metrics + params "
    "only). No-op without --mlflow.",
)
@click.pass_context
def cross_validation(
    ctx: click.Context,
    directory: Path,
    behavior: str,
    k: int,
    grouping_strategy: str | None,
    grouping_pattern: str | None,
    classifier: str,
    report_file: Path | None,
    mlflow_env: str | None,
    mlflow_experiment: str | None,
    mlflow_tags: tuple[str, ...],
    mlflow_no_report: bool,
):
    """Run leave-one-group-out cross-validation for a JABS project."""
    if report_file is not None and report_file.suffix.lower() not in {".md", ".json"}:
        raise click.ClickException(
            "Report file must have a .md (Markdown) or .json (JSON) extension."
        )

    cv_grouping_by_name = {
        "video": CrossValidationGroupingStrategy.VIDEO,
        "individual": CrossValidationGroupingStrategy.INDIVIDUAL,
        "filename": CrossValidationGroupingStrategy.FILENAME_PATTERN,
    }
    cv_grouping = cv_grouping_by_name[grouping_strategy.lower()] if grouping_strategy else None

    # --mlflow: absent -> None (disabled); bare flag -> "" (ambient env);
    # with a path -> that .env file.
    mlflow_enabled = mlflow_env is not None
    mlflow_env_file: Path | None = None
    parsed_mlflow_tags: dict[str, str] = {}

    # If MLflow logging was requested but the optional 'mlflow' extra is not
    # installed, warn and ignore the MLflow options rather than failing -- the
    # cross-validation still runs and the report is still produced.
    if mlflow_enabled and not mlflow_available():
        click.echo(
            "Warning: MLflow logging was requested (--mlflow) but the optional 'mlflow' "
            "dependency is not installed; ignoring MLflow options. Install it with "
            "\"pip install 'jabs-behavior-classifier[mlflow]'\" to enable logging.",
            err=True,
        )
        mlflow_enabled = False

    # Only interpret the other MLflow options when logging is actually enabled.
    # They are documented as no-ops without --mlflow, so e.g. a malformed
    # --mlflow-tag is ignored rather than failing the command.
    if mlflow_enabled:
        mlflow_env_file = Path(mlflow_env) if mlflow_env else None
        try:
            parsed_mlflow_tags = parse_kv_tags(list(mlflow_tags))
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    try:
        classifier_type = ClassifierType[classifier.upper()]
        run_cross_validation(
            directory,
            behavior,
            classifier_type,
            cv_grouping,
            k,
            report_file,
            grouping_regex=grouping_pattern,
            mlflow_enabled=mlflow_enabled,
            mlflow_env_file=mlflow_env_file,
            mlflow_experiment=mlflow_experiment,
            mlflow_tags=parsed_mlflow_tags,
            mlflow_log_report=not mlflow_no_report,
        )
    except MlflowLoggingError:
        # Cross-validation and the report succeeded; only the optional MLflow
        # push failed (already reported on the console). Use a distinct, non-zero
        # exit code so automation can tell this apart from a CV failure.
        ctx.exit(3)
    except Exception as e:
        raise click.ClickException(str(e)) from e


@cli.command(name="convert-to-nwb")
@click.argument(
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--multisubject",
    is_flag=True,
    default=False,
    help=(
        "Write a single multi-subject NWB file (using the ndx-multisubjects "
        "extension) instead of the default one file per identity. The combined "
        "file is written directly to OUTPUT."
    ),
)
@click.option(
    "--session-description",
    type=str,
    default=None,
    help="NWB session description string. Defaults to 'JABS PoseEstimation Data'.",
)
@click.option(
    "--subjects",
    "subjects_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to a JSON file containing per-animal biological metadata. "
        "Keys are identity names: use the external IDs from the pose file "
        "if present (e.g. 'mouse_a'), or 'subject_0', 'subject_1', … if the "
        "pose file has no external IDs. "
        "DANDI requires species, sex, and age (ISO 8601 duration, e.g. 'P70D') "
        "or date_of_birth (ISO 8601 datetime) on every subject. "
        "Additional fields: subject_id, genotype, strain, weight, description."
    ),
)
@click.option(
    "--session-metadata",
    "session_metadata_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to a JSON file containing NWB session-level metadata. "
        "Supported keys: session_start_time (ISO 8601 string), "
        "experimenter (string or list of strings), "
        "lab, institution, experiment_description, session_id (strings)."
    ),
)
@click.pass_context
def convert_to_nwb(
    ctx: click.Context,
    input_path: Path,
    output: Path,
    multisubject: bool,
    session_description: str | None,
    subjects_path: Path | None,
    session_metadata_path: Path | None,
) -> None:
    """Convert a JABS pose estimation file to NWB format.

    INPUT_PATH is a JABS pose HDF5 file (any version, v2-v8). The format
    version is inferred automatically from the filename (e.g. _pose_est_v6.h5).

    OUTPUT is the destination NWB file. By default one file per identity is
    written: OUTPUT is a naming template and is not created directly; instead
    one file per identity is written as {output_stem}_{identity_name}.nwb in the
    same directory. With --multisubject, a single combined file is written
    directly to OUTPUT.

    Examples:

    \b
        # One NWB file per identity (default)
        jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb

    \b
        # A single multi-subject file (ndx-multisubjects)
        jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb --multisubject

    \b
        # Include per-animal metadata
        jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb --subjects subjects.json

    \b
        # Specify session start time and other session metadata
        jabs-cli convert-to-nwb session_pose_est_v6.h5 session.nwb --session-metadata session.json
    """
    if ctx.obj["VERBOSE"]:
        click.echo(f"Input:  {input_path}")
        click.echo(f"Output: {output}")
        click.echo(f"Multisubject: {multisubject}")
        if subjects_path:
            click.echo(f"Subjects: {subjects_path}")
        if session_metadata_path:
            click.echo(f"Session metadata: {session_metadata_path}")

    subjects: dict[str, dict] | None = None
    if subjects_path is not None:
        try:
            subjects = json.loads(subjects_path.read_text())
        except Exception as e:
            raise click.ClickException(f"Failed to read subjects file: {e}") from e
        if not isinstance(subjects, dict):
            raise click.ClickException(
                f"Subjects file must contain a JSON object, got {type(subjects).__name__}"
            )

    session_metadata: dict | None = None
    if session_metadata_path is not None:
        try:
            session_metadata = json.loads(session_metadata_path.read_text())
        except Exception as e:
            raise click.ClickException(f"Failed to read session metadata file: {e}") from e
        if not isinstance(session_metadata, dict):
            raise click.ClickException(
                "Session metadata file must contain a JSON object, "
                f"got {type(session_metadata).__name__}"
            )

    console = Console()
    with console.status(f"Converting {input_path.name} → NWB ...", spinner="dots"):
        try:
            run_conversion(
                input_path=input_path,
                output_path=output,
                multisubject=multisubject,
                session_description=session_description,
                subjects=subjects,
                session_metadata=session_metadata,
            )
        except Exception as e:
            raise click.ClickException(str(e)) from e

    if multisubject:
        click.echo(f"Wrote {output}")
    else:
        click.echo(f"Wrote per-identity NWB files to {output.parent}")


def main():
    """Entry point for the JABS CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
