import cv2
import numpy as np

from jabs.pose_estimation import PoseEstimation, PoseEstimationV6

_ID_COLOR = (215, 222, 0)
_ACTIVE_COLOR = (0, 0, 255)

_FUTURE_TRACK_COLOR = (61, 61, 255)
_PAST_TRACK_COLOR = (135, 135, 255)

_CORNER_COLOR = (0, 135, 252)
_LIXIT_COLOR = (215, 222, 0)
_HOPPER_COLOR = (0, 255, 0)


__CONNECTED_SEGMENTS = [
    [
        PoseEstimation.KeypointIndex.LEFT_FRONT_PAW,
        PoseEstimation.KeypointIndex.CENTER_SPINE,
        PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW,
    ],
    [
        PoseEstimation.KeypointIndex.LEFT_REAR_PAW,
        PoseEstimation.KeypointIndex.BASE_TAIL,
        PoseEstimation.KeypointIndex.RIGHT_REAR_PAW,
    ],
    [
        PoseEstimation.KeypointIndex.NOSE,
        PoseEstimation.KeypointIndex.BASE_NECK,
        PoseEstimation.KeypointIndex.CENTER_SPINE,
        PoseEstimation.KeypointIndex.BASE_TAIL,
        PoseEstimation.KeypointIndex.MID_TAIL,
        PoseEstimation.KeypointIndex.TIP_TAIL,
    ],
]


def __gen_line_fragments(exclude_points: np.ndarray):
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
    for curr_pt_indexes in __CONNECTED_SEGMENTS:
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


def label_identity(
    img: np.ndarray,
    pose_est: PoseEstimation,
    identity: int,
    frame_index: int,
    color: tuple[int, int, int] = _ID_COLOR,
):
    """label the identity on an image

    Args:
        img: image to label
        pose_est: pose estimations for this video
        identity: identity to label
        frame_index: index of frame to label
        color: color to use for label

    Returns:
        None
    """
    shape = pose_est.get_identity_convex_hulls(identity)[frame_index]

    if shape is not None:
        center = shape.centroid

        # draw a marker at this location.
        cv2.circle(
            img,
            (int(center.x), int(center.y)),
            __scale_annotation_size(img, 4),
            color,
            -1,
            lineType=cv2.LINE_AA,
        )


def draw_track(
    img: np.ndarray,
    pose_est: PoseEstimation,
    identity: int,
    frame_index: int,
    future_points: int = 10,
    past_points: int = 5,
    point_index=PoseEstimation.KeypointIndex.NOSE,
):
    """draw a track for a specified identity

    Args:
        img: image to draw track on
        pose_est: pose estimations for this video
        identity: subject identity
        frame_index: index of the current frame
        future_points: number of frames after current frame to use for drawing the track
        past_points: number of frames before current frame to use for drawing the track
        point_index: index of pose key point to use for drawing track,
            if None use center of mass rather than a point. default to nose
    """
    slice_start = max(frame_index - past_points, 0)

    if point_index is None:
        convex_hulls = pose_est.get_identity_convex_hulls(identity)

        # get points for the 'future' track
        hulls = convex_hulls[frame_index : frame_index + future_points]
        centroids = [h.centroid for h in hulls if h is not None]
        future_track_points = [(int(c.x), int(c.y)) for c in centroids]

        # get points for 'past' track points
        hulls = convex_hulls[slice_start : frame_index + 1]
        centroids = [x.centroid for x in hulls if x is not None]
        past_track_points = [(int(x.x), int(x.y)) for x in centroids]

    else:
        # get points for 'future' track
        points, mask = pose_est.get_identity_poses(identity)
        future_track_points = points[frame_index : frame_index + future_points, point_index]
        track_point_mask = mask[frame_index : frame_index + future_points, point_index]
        # filter out masked points
        future_track_points = [(p[0], p[1]) for p in future_track_points[track_point_mask != 0]]

        # get points for 'past' track points
        past_track_points = points[slice_start : frame_index + 1, point_index]
        track_point_mask = mask[slice_start : frame_index + 1, point_index]
        past_track_points = [(p[0], p[1]) for p in past_track_points[track_point_mask != 0]]

    # draw circles at each future point
    for p in future_track_points:
        # draw a marker at this location.
        cv2.circle(
            img,
            (int(p[0]), int(p[1])),
            2,
            _FUTURE_TRACK_COLOR,
            -1,
            lineType=cv2.LINE_AA,
        )

    # draw line connecting points
    # convert to numpy array for opencv
    future_track_points = np.asarray(future_track_points, dtype=np.int32)
    cv2.polylines(img, [future_track_points], False, _FUTURE_TRACK_COLOR, 1)

    # draw circles at each past point
    for p in past_track_points:
        # draw a marker at this location.
        cv2.circle(img, (int(p[0]), int(p[1])), 2, _PAST_TRACK_COLOR, -1, lineType=cv2.LINE_AA)

    # draw line connecting points
    # convert to numpy array for opencv
    past_track_points = np.asarray(past_track_points, dtype=np.int32)
    cv2.polylines(img, [past_track_points], False, _PAST_TRACK_COLOR, 1)


def overlay_pose(img: np.ndarray, points: np.ndarray, mask: np.ndarray, color=(255, 255, 255)):
    """Overlay pose on a frame.

    Args:
        img: frame image
        points: pose points to overlay
        mask: points mask to indicate which points are valid
        color: color for overlay, defaults to white
    """
    if points is None:
        return

    # draw connections
    for seg in __gen_line_fragments(np.flatnonzero(mask == 0)):
        segment_points = [(p[0], p[1]) for p in points[seg]]
        # draw a wide black line
        cv2.polylines(
            img,
            [np.asarray(segment_points, dtype=np.int32)],
            False,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        # now draw a thin line with the specified color
        cv2.polylines(
            img,
            [np.asarray(segment_points, dtype=np.int32)],
            False,
            color,
            1,
            cv2.LINE_AA,
        )

    # draw points at each keypoint of the pose (if it exists at this frame)
    for point, point_mask in zip(points, mask, strict=True):
        if point_mask:
            cv2.circle(
                img,
                (int(point[0]), int(point[1])),
                __scale_annotation_size(img, 2),
                color,
                -1,
                lineType=cv2.LINE_AA,
            )


def trim_seg(arr: np.ndarray) -> np.ndarray | None:
    """Trims a single contour.  Returns an opencv-complaint contour (dtype = int).

    Args:
        arr: A numpy array with contour data.

    Returns:
        np.ndarray
    """
    assert arr.ndim == 2
    return_arr = arr[np.all(arr != -1, axis=1), :]
    if len(return_arr) > 0:
        return return_arr.astype(int)
    return None


def trim_seg_list(arr: np.ndarray) -> list:
    """Trims all contours for an individual.

    Args:
        arr: A numpy array with contour data.

    Returns:
        list
    """
    assert arr.ndim == 3
    return [trim_seg(x) for x in arr if np.any(x != -1)]


def draw_all_contours(img: np.ndarray, seg_data: np.ndarray, color: tuple[int, int, int]):
    """Draw all contours given data for a particular mouse in a particular video frame.

    Args:
        img: The current video frame.
        seg_data: This will be the segmentation for a particular frame
            and identity.
        color: color of segmentation contours rendered on the GUI.

    Returns:
        None
    """
    trimmed_contours = trim_seg_list(seg_data)
    cv2.drawContours(img, trimmed_contours, -1, color, 2)


def overlay_segmentation(
    img: np.ndarray, pose_est: PoseEstimationV6, identity: int, frame_index: int
):
    """overlay the segmentation on a frame fo ra given identity

    Args:
        img: The current video frame.
        pose_est: This will be a pose estimation object >= v6.
        identity: This integer identifies which mouse the segmentation
            will be applied to.
        frame_index: This integer identifies the current video frame
            index.

    Returns:
        None
    """
    # No segmentation data to display
    if pose_est.format_major_version < 6:
        return

    contours = pose_est.get_segmentation_data_per_frame(frame_index, identity)

    if contours is None:
        # No segmentation data available to render.
        return

    draw_all_contours(img, contours, _ACTIVE_COLOR)


def overlay_landmarks(img: np.ndarray, pose_est: PoseEstimation):
    """overlay landmarks on a frame"""
    static_objects = pose_est.static_objects

    # label arena corners if they were included in the pose file
    corners = static_objects.get("corners")
    if corners is not None:
        for i in range(4):
            cv2.circle(
                img,
                (int(corners[i, 0]), int(corners[i, 1])),
                2,
                _CORNER_COLOR,
                -1,
                lineType=cv2.LINE_AA,
            )

    # label lixit(s) if present in pose file
    # supports lixit with 3 keypoints or older style single keypoint
    lixit = pose_est.static_objects.get("lixit")
    if lixit is not None:
        lixit_keypoints = pose_est.num_lixit_keypoints

        # lixit is either # lixit x 2 or # lixit x 3 x 2
        # iterate over each lixit
        for i in range(lixit.shape[0]):
            if lixit_keypoints == 3:
                # lixit is 3 keypoint version, draw each keypoint
                for j in range(3):
                    pts = (int(lixit[i, j, 0]), int(lixit[i, j, 1]))
                    cv2.circle(
                        img,
                        pts,
                        __scale_annotation_size(img, 2),
                        _LIXIT_COLOR,
                        -1,
                        lineType=cv2.LINE_AA,
                    )
            else:
                # assume older style single keypoint lixit
                pts = (int(lixit[i, 0]), int(lixit[i, 1]))
                cv2.circle(
                    img,
                    pts,
                    __scale_annotation_size(img, 2),
                    _LIXIT_COLOR,
                    -1,
                    lineType=cv2.LINE_AA,
                )

    # draw food hopper if present in pose file
    hopper_points = pose_est.static_objects.get("food_hopper")
    if hopper_points is not None:
        cv2.polylines(img, np.int32([hopper_points]), True, _HOPPER_COLOR, 1, lineType=cv2.LINE_AA)


def __scale_annotation_size(img: np.ndarray, size: int | float) -> int | float:
    """Scale the size of the landmark markers based on the size of the image.

    800x800 is the video size jabs was developed with, so we use that as a reference

    Args:
        img: image
        size: unscaled size
    """
    if type(size) is int:
        # scale an integer size, add half of the baseline image size so it effectively rounds up
        return (img.shape[0] + 400) // 800 * size
    elif type(size) is float:
        return img.shape[0] / 800.0 * size
    else:
        raise ValueError("size must be int or float")
