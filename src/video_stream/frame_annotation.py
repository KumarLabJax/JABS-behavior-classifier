import cv2
from shapely.geometry import MultiPoint
from itertools import compress

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

        # first remove any invalid points (where mask is not True)
        filtered_points = list(compress(points[:-2], mask[:-2]))

        # find the center of the remaining points
        center = MultiPoint(filtered_points).convex_hull.centroid

        # draw a marker at this location. this is a filled in circle and then
        # a larger unfilled circle
        cv2.circle(img, (int(center.y), int(center.x)), 3, _FRAME_LABEL_COLOR,
                   -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(center.y), int(center.x)), 6, _FRAME_LABEL_COLOR,
                   1, lineType=cv2.LINE_AA)

