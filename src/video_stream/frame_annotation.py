import cv2
from shapely.geometry import MultiPoint

_FRAME_LABEL_COLOR = (215, 222, 0)


def label_identity(img, points, mask):
    """
    label the identity on an image
    :param img: image to label
    :param points: array of 12 keypoints that define the pose
    :param mask: mask that indicates if each point is valid or not (sometimes
    pose will be missing points)
    :return: None
    """
    if points is not None:
        # note, we only use the first 10 points (out of 12). We are ignoring
        # the mid tail and the tip of the tail

        # first remove any invalid points (where mask is not 1)
        filtered_points = points[:-2][mask[:-2] == 1]

        # find the center of the remaining points
        center = MultiPoint(filtered_points).convex_hull.centroid

        # draw a marker at this location. this is a filled in circle and then
        # a larger unfilled circle
        cv2.circle(img, (int(center.y), int(center.x)), 3, _FRAME_LABEL_COLOR,
                   -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(center.y), int(center.x)), 6, _FRAME_LABEL_COLOR,
                   1, lineType=cv2.LINE_AA)


def label_all_identities(img, pose_est, identities, frame_index):
    """
    label all of the identities in the frame
    :param img: image to draw the labels on
    :param pose_est: pose estimations for this video
    :param identities: list of identity names
    :param frame_index: index of frame, used to get all poses for frame
    :return: None
    """

    for identity in identities:
        points, mask = pose_est.get_points(frame_index, identity)
        if points is not None:
            # first remove any invalid points (where mask is not 1)
            filtered_points = points[:-2][mask[:-2] == 1]

            # find the center of the remaining points
            center = MultiPoint(filtered_points).convex_hull.centroid

            # write the identity at that location
            cv2.putText(img, str(identity), (int(center.y), int(center.x)),
                        cv2.FONT_HERSHEY_PLAIN, 1, _FRAME_LABEL_COLOR, 1,
                        lineType=cv2.LINE_AA)

