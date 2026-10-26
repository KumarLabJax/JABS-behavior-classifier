"""Tests for the shared detail-bar geometry helpers in jabs.ui."""

import numpy as np
import pytest

try:
    from jabs.ui.behavior_timeline.track_widgets.timeline_util import (
        frame_pixel_widths,
        window_geometry,
    )

    SKIP_UI_TESTS = False
except ImportError as e:
    SKIP_UI_TESTS = True
    SKIP_REASON = f"Qt/UI dependencies not available: {e}"

pytestmark = pytest.mark.skipif(
    SKIP_UI_TESTS,
    reason=SKIP_REASON if SKIP_UI_TESTS else "",
)


def test_window_geometry_fully_in_bounds() -> None:
    """A window well inside the video has no padding and spans the full width."""
    geometry = window_geometry(
        current_frame=500, window_size=100, num_frames=1000, frame_width=2.0
    )

    assert (geometry.start, geometry.end) == (400, 600)
    assert (geometry.slice_start, geometry.slice_end) == (400, 600)
    assert geometry.n_in_bounds == 201
    assert geometry.n_start_pad == 0
    assert (geometry.x0, geometry.width_px) == (0, 402)
    assert geometry.has_in_bounds_content


def test_window_geometry_pads_before_video_start() -> None:
    """A window centered near frame zero pads the frames before the video."""
    geometry = window_geometry(current_frame=10, window_size=100, num_frames=1000, frame_width=2.0)

    assert geometry.start == -90
    assert geometry.n_start_pad == 90
    assert (geometry.slice_start, geometry.slice_end) == (0, 110)
    assert geometry.n_in_bounds == 111
    assert (geometry.x0, geometry.width_px) == (180, 222)


def test_window_geometry_clamps_to_last_frame() -> None:
    """A window centered near the end stops at the video's last frame."""
    geometry = window_geometry(
        current_frame=990, window_size=100, num_frames=1000, frame_width=2.0
    )

    assert geometry.end == 1090
    assert geometry.slice_end == 999
    assert geometry.n_in_bounds == 110
    assert geometry.n_start_pad == 0
    assert (geometry.x0, geometry.width_px) == (0, 220)


def test_window_geometry_no_frames() -> None:
    """A video with no frames yields no in-bounds content."""
    geometry = window_geometry(current_frame=0, window_size=100, num_frames=0, frame_width=2.0)

    assert geometry.n_in_bounds <= 0
    assert not geometry.has_in_bounds_content


def test_window_geometry_subpixel_frames_have_no_content() -> None:
    """A window narrower than one pixel per frame can still occupy zero pixels."""
    geometry = window_geometry(current_frame=0, window_size=100, num_frames=1, frame_width=0.4)

    assert geometry.n_in_bounds == 1
    assert geometry.width_px == 0
    assert not geometry.has_in_bounds_content


@pytest.mark.parametrize(
    ("frame_width", "num_frames"),
    [(2.0, 1000), (0.37, 1000), (1.0, 1000), (3.25, 1000), (0.5, 211)],
    ids=["wide", "subpixel", "exact", "fractional", "half"],
)
def test_frame_pixel_widths_tile_the_region(frame_width: float, num_frames: int) -> None:
    """Per-frame widths cover the in-bounds region exactly, with no gap or overlap."""
    geometry = window_geometry(
        current_frame=50, window_size=100, num_frames=num_frames, frame_width=frame_width
    )
    widths = frame_pixel_widths(geometry, frame_width)

    assert widths.shape == (geometry.n_in_bounds,)
    assert np.all(widths >= 0)
    assert widths.sum() == geometry.width_px


def test_frame_pixel_widths_uniform_when_frame_width_is_integral() -> None:
    """An integral pixels-per-frame gives every frame the same width."""
    geometry = window_geometry(
        current_frame=500, window_size=100, num_frames=1000, frame_width=3.0
    )

    assert np.all(frame_pixel_widths(geometry, 3.0) == 3)
