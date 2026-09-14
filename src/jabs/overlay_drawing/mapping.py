"""Helpers for mapping image-space points into a painter's coordinate space.

An overlay maps each point through a caller-supplied ``to_output``, which returns
``None`` for a point the display crop excludes. Anything drawn as a connected series
of points then has to be split at those gaps: joining across one would draw a line
through the region that was cropped away.
"""

from collections.abc import Iterator


def visible_runs(
    mapped: list[tuple[int, int] | None],
) -> Iterator[tuple[list[tuple[int, int]], bool]]:
    """Split mapped points into runs of consecutive visible ones.

    Args:
        mapped: Points already mapped to the painter's space, with ``None`` for each
            point that was dropped.

    Yields:
        ``(run, complete)`` for each non-empty run, where ``complete`` says the run is
        the whole sequence with nothing dropped. Callers that close a shape need that
        flag; callers drawing an open line can ignore it. Single-point runs are
        included, so a caller that cannot draw one has to skip it itself.
    """
    run: list[tuple[int, int]] = []
    dropped = False
    for point in mapped:
        if point is None:
            dropped = True
            if run:
                yield run, False
            run = []
            continue
        run.append(point)
    if run:
        yield run, not dropped
