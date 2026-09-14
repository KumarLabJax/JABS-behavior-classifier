"""Helpers for mapping image-space points into a painter's coordinate space.

An overlay maps each point through a caller-supplied ``to_output``, which returns
``None`` for a point the display crop excludes. Anything drawn as a connected series
of points then has to be split at those gaps: joining across one would draw a line
through the region that was cropped away.
"""

from collections.abc import Iterator


def visible_runs(
    mapped: list[tuple[int, int] | None],
    *,
    closed: bool = False,
) -> Iterator[tuple[list[tuple[int, int]], bool]]:
    """Split mapped points into runs of consecutive visible ones.

    Args:
        mapped: Points already mapped to the painter's space, with ``None`` for each
            point that was dropped.
        closed: Whether the points describe a closed shape, where the last point joins
            back to the first. That edge is part of the shape, so the points are
            walked as a ring: the sequence is rotated to begin at a gap, which keeps a
            run that spans the wrap-around in one piece instead of splitting it into
            the head and tail of the list.

    Yields:
        ``(run, complete)`` for each non-empty run, where ``complete`` says the run is
        the whole sequence with nothing dropped. Callers that close a shape need that
        flag; callers drawing an open line can ignore it. Single-point runs are
        included, so a caller that cannot draw one has to skip it itself.
    """
    if closed and any(point is None for point in mapped):
        first_gap = next(i for i, point in enumerate(mapped) if point is None)
        mapped = mapped[first_gap:] + mapped[:first_gap]

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
