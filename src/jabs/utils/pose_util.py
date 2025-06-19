from collections.abc import Generator

import numpy as np

from jabs.pose_estimation import PoseEstimation


def gen_line_fragments(exclude_points: np.ndarray) -> Generator[list[int], None, None]:
    """generate line fragments from the connected segments.

    This will break up segments if a point within the segment is excluded,
    or will remove the segment completely if it does not have at least two points

    Args:
        exclude_points: list of points to exclude when generating
            segments

    Returns:
        yields lists of Keypoint indexes that make up the segments to draw
    """
    curr_fragment = []
    for curr_pt_indexes in PoseEstimation.CONNECTED_SEGMENTS:
        for curr_pt_index in curr_pt_indexes:
            if curr_pt_index.value in exclude_points:
                if len(curr_fragment) >= 2:
                    yield curr_fragment
                curr_fragment = []
            else:
                curr_fragment.append(curr_pt_index.value)
        if len(curr_fragment) >= 2:
            yield curr_fragment
        curr_fragment = []
