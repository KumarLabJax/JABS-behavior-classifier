"""Compute and cache JABS features for a pose file."""

import logging
from pathlib import Path

import click
from rich.console import Console

from jabs.core.enums import CacheFormat, ProjectDistanceUnit
from jabs.feature_extraction.features import IdentityFeatures
from jabs.pose_estimation import PoseOpenError, open_pose_file
from jabs.pose_estimation.discovery import is_pose_file
from jabs.project import Project

logger = logging.getLogger(__name__)


@click.command(name="compute-features")
@click.option(
    "--pose-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pose file to compute features for.",
)
@click.option(
    "--feature-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write output features.",
)
@click.option(
    "--use-pixel-distances",
    "use_pixel_distances",
    is_flag=True,
    help="Force pixel distance units (overrides the cm default when the pose file supports it).",
)
@click.option(
    "-w",
    "--window-size",
    "window_sizes",
    type=click.IntRange(min=1),
    multiple=True,
    help=(
        "Window size for features. Repeat to compute multiple window sizes "
        "(e.g. -w 5 -w 10). Omit to skip window feature computation."
    ),
)
@click.option(
    "--fps",
    type=click.IntRange(min=1),
    default=None,
    help="Deprecated and ignored: the frame rate is read from the pose file.",
)
@click.option(
    "--use-pose-hash",
    is_flag=True,
    default=False,
    help=(
        "Include the pose file hash as a subdirectory level in the feature cache path "
        "(e.g. <feature-dir>/<video>/<pose-hash>/<identity>); "
        "prevents collisions when a shared cache dir is used across multiple pipelines."
    ),
)
@click.option(
    "--cache-format",
    type=click.Choice([f.value for f in CacheFormat], case_sensitive=False),
    default=CacheFormat.PARQUET.value,
    show_default=True,
    help="Storage format for the feature cache.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Recompute features and overwrite the cache even when a valid cache exists.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory to keep conversions of legacy *_pose_est_v<N>.h5 pose files in, so "
        "later runs reuse them. By default a legacy file is converted into a temporary "
        "directory removed when the command finishes."
    ),
)
def compute_features_command(
    pose_file: Path,
    feature_dir: Path,
    use_pixel_distances: bool,
    window_sizes: tuple[int, ...],
    fps: int | None,
    use_pose_hash: bool,
    cache_format: str,
    force: bool,
    cache_dir: Path | None,
) -> None:
    """Compute and cache JABS features for a pose file.

    The pose file is a unified ``*_pose.h5`` or a legacy ``*_pose_est_v<N>.h5``
    file. The features computed follow what the file contains (its capabilities
    and static objects), and time-derived features use the frame rate it records.
    Distance units default to cm when the pose file provides a pixel-to-cm
    scale, otherwise pixel; use --use-pixel-distances to force pixel units.
    """
    if fps is not None:
        logger.warning("--fps is deprecated and ignored; the pose file's frame rate is used")

    if not is_pose_file(pose_file):
        raise click.ClickException(
            f"'{pose_file.name}' is not a pose file name; expected '*_pose.h5' or "
            "'*_pose_est_v<N>.h5'."
        )

    console = Console()
    with console.status(f"Opening pose file {pose_file.name}...", spinner="dots"):
        try:
            pose_est = open_pose_file(pose_file, cache_dir, temporary_conversion=cache_dir is None)
        except PoseOpenError as e:
            raise click.ClickException(str(e)) from e

    # the pose stays open while features are computed; close it when done
    with pose_est:
        if use_pixel_distances or pose_est.cm_per_pixel is None:
            distance_unit = ProjectDistanceUnit.PIXEL
        else:
            distance_unit = ProjectDistanceUnit.CM

        capabilities = pose_est.capabilities
        settings = Project.settings_by_capabilities(capabilities, distance_unit=distance_unit)

        sorted_window_sizes = sorted(set(window_sizes))
        cache_window = bool(sorted_window_sizes)

        cache_format_enum = CacheFormat(cache_format.lower())

        logger.info(
            "computing features for %s (capabilities %s, %s, cache=%s)",
            pose_file,
            sorted(c.value for c in capabilities),
            distance_unit.name,
            cache_format_enum.value,
        )

        identities = list(pose_est.identities)
        with console.status("Computing features...", spinner="dots") as status:
            for curr_id in identities:
                status.update(
                    f"Computing per-frame features for identity {curr_id} "
                    f"({identities.index(curr_id) + 1}/{len(identities)})..."
                )
                # Note: every feature the pose file supports is cached; filtering by
                # the settings happens when get_features is called.
                features = IdentityFeatures(
                    pose_file,
                    curr_id,
                    feature_dir,
                    pose_est,
                    force=force,
                    op_settings=settings,
                    cache_window=cache_window,
                    cache_format=cache_format_enum,
                    include_pose_hash=use_pose_hash,
                )
                for ws in sorted_window_sizes:
                    status.update(
                        f"Computing window features (size={ws}) for identity {curr_id} "
                        f"({identities.index(curr_id) + 1}/{len(identities)})..."
                    )
                    _ = features.get_window_features(ws, force=force)
