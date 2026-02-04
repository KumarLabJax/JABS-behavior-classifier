"""Postprocessing stages."""

from .duration_stage import BoutDurationFilterStage
from .interpolation_stage import GapInterpolationStage
from .postprocessing_stage import PostprocessingStage
from .stitching_stage import BoutStitchingStage

__all__ = [
    "BoutDurationFilterStage",
    "BoutStitchingStage",
    "GapInterpolationStage",
    "PostprocessingStage",
    "stage_registry",
]

# map a stage name string to the class type of the stage implementation
_STAGE_REGISTRY: dict[str, type[PostprocessingStage]] = {
    BoutDurationFilterStage.name: BoutDurationFilterStage,
    GapInterpolationStage.name: GapInterpolationStage,
    BoutStitchingStage.name: BoutStitchingStage,
}


def stage_registry() -> dict[str, type[PostprocessingStage]]:
    """Get the stage registry.

    Returns:
        Dictionary mapping stage names to stage classes.
    """
    return _STAGE_REGISTRY
