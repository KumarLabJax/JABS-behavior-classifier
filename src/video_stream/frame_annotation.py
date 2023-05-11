import cv2
import numpy as np
from typing import Tuple, List
from src.pose_estimation import PoseEstimation

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
        PoseEstimation.KeypointIndex.RIGHT_FRONT_PAW
    ],
    [
        PoseEstimation.KeypointIndex.LEFT_REAR_PAW,
        PoseEstimation.KeypointIndex.BASE_TAIL,
        PoseEstimation.KeypointIndex.RIGHT_REAR_PAW
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


def __gen_line_fragments(exclude_points):
    """
    generate line fragments from the connected segments. will break up
    segments if a point within the segment is excluded, or will remove the
    segment completely if it does not have at least two points
    :param exclude_points: list of points to exclude when generating segments
    :return: yields lists of Keypoint indexes that make up the segments to draw
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


def label_identity(img, pose_est, identity, frame_index,
                   color=_ID_COLOR):
    """
    label the identity on an image
    :param img: image to label
    :param pose_est: pose estimations for this video
    :param identity: identity to label
    :param frame_index: index of frame to label
    :param color: color to use for label
    If point = None, use center of mass.
    :return: None
    """

    shape = pose_est.get_identity_convex_hulls(identity)[frame_index]

    if shape is not None:
        center = shape.centroid

        # draw a marker at this location.
        cv2.circle(img, (int(center.y), int(center.x)), 2, color,
                   -1, lineType=cv2.LINE_AA)


def label_all_identities(img, pose_est, identities, frame_index, subject=None):
    """
    label all of the identities in the frame
    :param img: image to draw the labels on
    :param pose_est: pose estimations for this video
    :param identities: list of identity names
    :param frame_index: index of frame, used to get all poses for frame
    :param subject: identity to label as 'subject'
    :return: None
    """

    for identity in identities:
        shape = pose_est.get_identity_convex_hulls(identity)[frame_index]
        if shape is not None:
            center = shape.centroid

            if identity == subject:
                color = _ACTIVE_COLOR
            else:
                color = _ID_COLOR
            # write the identity at that location
            cv2.putText(img, str(identity), (int(center.y), int(center.x)),
                        cv2.FONT_HERSHEY_PLAIN, 1.25, color, 2,
                        lineType=cv2.LINE_AA)


def draw_track(img: np.ndarray, pose_est: PoseEstimation, identity: int,
               frame_index: int, future_points: int = 10, past_points: int = 5,
               point_index=PoseEstimation.KeypointIndex.NOSE):
    """
    draw a track for a specified identity

    :param img: image to draw track on
    :param pose_est: pose estimations for this video
    :param identity: subject identity
    :param frame_index: index of the current frame
    :param future_points: number of frames after current frame to use
    for drawing the track
    :param past_points: number of frames before current frame to use for
    drawing the track
    :param point_index: index of pose key point to use for drawing track,
    if None use center of mass rather than a point. default to nose
    """

    slice_start = max(frame_index-past_points, 0)

    if point_index is None:
        convex_hulls = pose_est.get_identity_convex_hulls(identity)

        # get points for the 'future' track
        hulls = convex_hulls[frame_index:frame_index + future_points]
        centroids = [h.centroid for h in hulls if h is not None]
        # openCV needs points to be ordered y,x
        future_track_points = [(int(c.y), int(c.x)) for c in centroids]

        # get points for 'past' track points
        hulls = convex_hulls[slice_start:frame_index+1]
        centroids = [x.centroid for x in hulls if x is not None]
        # openCV needs points to be ordered y,x
        past_track_points = [(int(x.y), int(x.x)) for x in centroids]

    else:
        # get points for 'future' track
        points, mask = pose_est.get_identity_poses(identity)
        future_track_points = points[frame_index:frame_index+future_points,
                                     point_index]
        track_point_mask = mask[frame_index:frame_index+future_points, point_index]
        # filter out masked out points, reorder point as y,x for openCV
        future_track_points = [
            (p[1], p[0]) for p in future_track_points[track_point_mask != 0]
        ]

        # get points for 'past' track points
        past_track_points = points[slice_start:frame_index+1, point_index]
        track_point_mask = mask[slice_start:frame_index+1, point_index]
        past_track_points = [
            (p[1], p[0]) for p in past_track_points[track_point_mask != 0]
        ]

    # draw circles at each future point
    for p in future_track_points:
        # draw a marker at this location.
        cv2.circle(img, (p[0], p[1]), 2, _FUTURE_TRACK_COLOR,
                   -1, lineType=cv2.LINE_AA)

    # draw line connecting points
    # convert to numpy array for opencv
    future_track_points = np.asarray(future_track_points, dtype=np.int32)
    cv2.polylines(img, [future_track_points], False, _FUTURE_TRACK_COLOR, 1)

    # draw circles at each past point
    for p in past_track_points:
        # draw a marker at this location.
        cv2.circle(img, (p[0], p[1]), 2, _PAST_TRACK_COLOR,
                   -1, lineType=cv2.LINE_AA)

    # draw line connecting points
    # convert to numpy array for opencv
    past_track_points = np.asarray(past_track_points, dtype=np.int32)
    cv2.polylines(img, [past_track_points], False, _PAST_TRACK_COLOR, 1)


def overlay_pose(img: np.ndarray, points: np.ndarray, mask: np.ndarray,
                 color=(255, 255, 255)):
    """

    :param img:
    :param points:
    :param mask:
    :param color:
    :return:
    """

    if points is None:
        return

    # draw connections
    for seg in __gen_line_fragments(np.flatnonzero(mask==0)):
        segment_points = [(p[1], p[0]) for p in points[seg]]
        # draw a wide black line
        cv2.polylines(img, [np.asarray(segment_points, dtype=np.int32)], False,
                      (0, 0, 0), 2, cv2.LINE_AA)
        # now draw a thin line with the specified color
        cv2.polylines(img, [np.asarray(segment_points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)

    # draw points at each keypoint of the pose (if it exists at this frame)
    for point, point_mask in zip(points, mask):
        if point_mask:
            cv2.circle(img, (point[1], point[0]), 2, color,
                       -1, lineType=cv2.LINE_AA)


def trim_seg(arr: np.ndarray) -> np.ndarray:
    """
    Trims a single contour.  Returns an opencv-complaint contour (dtype = int).

    :param array: A numpy array with contour data.
    :return: np.ndarray
    """
    assert arr.ndim == 2
    return_arr = arr[np.all(arr!=-1, axis=1),:]
    if len(return_arr)>0:
        return return_arr.astype(int)


def trim_seg_list(arr: np.ndarray) -> List:
    """
    Trims all contours for an individual.

    :param array: A numpy array with contour data.
    :return: List
    """
    assert arr.ndim == 3
    return [trim_seg(x) for x in arr if np.any(x!=-1)]


def draw_all_contours(img: np.ndarray, seg_data: np.ndarray, color: Tuple[int, int, int]):
    """
    Draw all contours given data for a particular mouse in a particular video frame.

    :param img: The current video frame.
    :param pose_est: This will be a pose estimation object >= v6.
    :param identity: This integer identifies which mouse the segmentation will be applied to.
    :param frameIndex: This integer identifies the current video frame index.
    :param color [optional]: color of segmentation contours rendered on the GUI.
    :return: None
    """
    trimmed_contours = trim_seg_list(seg_data)
    cv2.drawContours(img, trimmed_contours, -1, color, 2)
        

def overlay_segmentation(img: np.ndarray, pose_est: PoseEstimation,
    identity: int, frameIndex: int, identities=None, color=(255, 255, 255)):
    """
    :param img: The current video frame.
    :param pose_est: This will be a pose estimation object >= v6.
    :param identity: This integer identifies which mouse the segmentation will be applied to.
    :param frameIndex: This integer identifies the current video frame index.
    :param color [optional]: color of segmentation contours rendered on the GUI.
    :return: None
    """

    contours = pose_est.get_segmentation_data_per_frame(frameIndex, identity)
    
    if contours is None:
        # No segmentation data available to render.
        return
    
    draw_all_contours(img, contours, _ACTIVE_COLOR)

    # Old way (< 5/11/2023)
    if False:
        # Discard values < 0.
        contours = contours[contours >= 0]
        if len(contours) > 0: 
            # Filter missing contours and coerce type to satisfy drawContours
            # method.
            contours = contours.reshape((len(contours)//2, 2)).astype(int)
            cv2.drawContours(img, [contours], -1, _ACTIVE_COLOR, 2)


def overlay_landmarks(img: np.ndarray, pose_est: PoseEstimation):
    static_objects = pose_est.static_objects

    # draw a marker at this location.
    corners = static_objects.get('corners')
    if corners is not None:
        for i in range(4):
            cv2.circle(img, (corners[i, 0], corners[i, 1]), 2, _CORNER_COLOR,
                       -1, lineType=cv2.LINE_AA)

    lixit = pose_est.static_objects.get('lixit')
    if lixit is not None:
        for i in range(lixit.shape[0]):
            x, y = lixit[i][0], lixit[i][1]
            cv2.circle(img, (int(y), int(x)), 2, _LIXIT_COLOR,
                       -1, lineType=cv2.LINE_AA)

    hopper_points = pose_est.static_objects.get('food_hopper')
    if hopper_points is not None:
        hopper = [(p[1], p[0]) for p in hopper_points]
        cv2.polylines(img, np.int32([hopper]), True, _HOPPER_COLOR,
                      1, lineType=cv2.LINE_AA)
