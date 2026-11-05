"""Tests for the shared detail-bar geometry helpers in jabs.ui."""

from unittest import mock

import numpy as np
import pytest

from jabs.behavior_search import SearchHit

try:
    from jabs.ui.behavior_timeline.track_widgets.timeline_util import (
        frame_pixel_widths,
        render_search_hits,
        window_frame_count,
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


@pytest.mark.parametrize(("window_size", "expected"), [(0, 1), (1, 3), (100, 201)])
def test_window_frame_count_includes_current_frame(window_size: int, expected: int) -> None:
    """A window spans its frames on both sides plus the current frame."""
    assert window_frame_count(window_size) == expected


# a 201 frame window (window size 100) starting at frame 400, as a detail bar paints it
_WINDOW_START = 400
_WINDOW_FRAMES_TOTAL = 201
_BAR_HEIGHT = 32
_FRAME_WIDTH = 2.0


def _hit(start_frame: int, end_frame: int) -> SearchHit:
    """Build a search hit covering an inclusive frame range."""
    return SearchHit(
        file="video.avi",
        identity="0",
        behavior="walking",
        start_frame=start_frame,
        end_frame=end_frame,
    )


def _render(hit: SearchHit) -> mock.Mock:
    """Render one hit onto a recording painter and return the painter."""
    painter = mock.Mock()
    render_search_hits(
        painter,
        [hit],
        0,
        _WINDOW_START,
        _FRAME_WIDTH,
        _BAR_HEIGHT,
        _WINDOW_FRAMES_TOTAL,
    )
    return painter


def _drew_anything(painter: mock.Mock) -> bool:
    """Whether the painter was asked to draw a hit marker or either end diamond."""
    return bool(painter.drawLine.call_args_list or painter.drawPolygon.call_args_list)


@pytest.mark.parametrize(
    ("start_frame", "end_frame"),
    [
        (_WINDOW_START - 10, _WINDOW_START - 1),
        (_WINDOW_START - 1, _WINDOW_START - 1),
        (_WINDOW_START + _WINDOW_FRAMES_TOTAL, _WINDOW_START + _WINDOW_FRAMES_TOTAL + 10),
        (_WINDOW_START + _WINDOW_FRAMES_TOTAL + 5, _WINDOW_START + _WINDOW_FRAMES_TOTAL + 5),
    ],
    ids=["before", "ends-one-frame-before", "starts-one-frame-after", "well-after"],
)
def test_render_search_hits_skips_hits_outside_the_window(
    start_frame: int, end_frame: int
) -> None:
    """A hit that does not overlap the window draws nothing, not a marker at the edge."""
    assert not _drew_anything(_render(_hit(start_frame, end_frame)))


@pytest.mark.parametrize(
    ("start_frame", "end_frame"),
    [
        (_WINDOW_START, _WINDOW_START),
        (_WINDOW_START - 10, _WINDOW_START),
        (_WINDOW_START + _WINDOW_FRAMES_TOTAL - 1, _WINDOW_START + _WINDOW_FRAMES_TOTAL + 10),
        (_WINDOW_START - 50, _WINDOW_START + _WINDOW_FRAMES_TOTAL + 50),
    ],
    ids=["first-frame", "overlaps-start", "last-frame", "spans-window"],
)
def test_render_search_hits_draws_hits_overlapping_the_window(
    start_frame: int, end_frame: int
) -> None:
    """A hit sharing even one frame with the window is drawn."""
    assert _drew_anything(_render(_hit(start_frame, end_frame)))
