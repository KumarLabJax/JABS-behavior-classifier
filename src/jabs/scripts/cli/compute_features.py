"""Compute and cache JABS features for a pose file."""

import logging
from pathlib import Path

import click

from jabs.core.enums import CacheFormat, ProjectDistanceUnit
from jabs.feature_extraction.features import IdentityFeatures
from jabs.pose_estimation import get_pose_file_major_version, open_pose_file
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
    default=30,
    show_default=True,
    help="Frames per second to use for feature calculation.",
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
def compute_features_command(
    pose_file: Path,
    feature_dir: Path,
    use_pixel_distances: bool,
    window_sizes: tuple[int, ...],
    fps: int,
    use_pose_hash: bool,
    cache_format: str,
) -> None:
    """Compute and cache JABS features for a pose file.

    The pose file version is inferred from the filename (e.g. *_pose_est_v6.h5).
    Distance units default to cm when the pose file provides a pixel-to-cm
    scale, otherwise pixel; use --use-pixel-distances to force pixel units.
    """
    try:
        pose_version = get_pose_file_major_version(pose_file)
    except (AttributeError, ValueError) as e:
        raise click.ClickException(
            f"Unable to determine pose version from filename '{pose_file.name}'; "
            "expected a name like '*_pose_est_v<N>.h5'."
        ) from e

    pose_est = open_pose_file(pose_file)

    if use_pixel_distances or pose_est.cm_per_pixel is None:
        distance_unit = ProjectDistanceUnit.PIXEL
    else:
        distance_unit = ProjectDistanceUnit.CM

    settings = Project.settings_by_pose_version(pose_version, distance_unit)

    sorted_window_sizes = sorted(set(window_sizes))
    cache_window = bool(sorted_window_sizes)

    cache_format_enum = CacheFormat(cache_format.lower())

    logger.info(
        "computing features for %s (pose v%s, %s, cache=%s)",
        pose_file,
        pose_version,
        distance_unit.name,
        cache_format_enum.value,
    )

    for curr_id in pose_est.identities:
        # Note: Features are still cached with the highest pose version.
        # It isn't until get_features is called that filtering occurs.
        features = IdentityFeatures(
            pose_file,
            curr_id,
            feature_dir,
            pose_est,
            fps=fps,
            op_settings=settings,
            cache_window=cache_window,
            cache_format=cache_format_enum,
            include_pose_hash=use_pose_hash,
        )
        for ws in sorted_window_sizes:
            _ = features.get_window_features(ws, force=True)
