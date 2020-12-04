import cv2
import numpy as np

from src.pose_estimation import PoseEstimation

_ID_COLOR = (215, 222, 0)
_ACTIVE_COLOR = (0, 0, 255)

_FUTURE_TRACK_COLOR = (61, 61, 255)
_PAST_TRACK_COLOR = (135, 135, 255)


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


def draw_track(img, pose_est: PoseEstimation, identity: int, frame_index: int,
               future_points: int=10, past_points: int=5,
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
        centroids = [x.centroid for x in hulls if x is not None]
        # openCV needs points to be ordered y,x
        future_track_points = [(int(x.y), int(x.x)) for x in centroids]

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
