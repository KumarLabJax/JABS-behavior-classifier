import cv2
from shapely.geometry import MultiPoint

_ID_COLOR = (215, 222, 0)
_ACTIVE_COLOR = (0, 0, 255)


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


def label_all_identities(img, pose_est, identities, frame_index, active=None):
    """
    label all of the identities in the frame
    :param img: image to draw the labels on
    :param pose_est: pose estimations for this video
    :param identities: list of identity names
    :param frame_index: index of frame, used to get all poses for frame
    :return: None
    """

    for identity in identities:
        shape = pose_est.get_identity_convex_hulls(identity)[frame_index]
        if shape is not None:
            center = shape.centroid

            if identity == active:
                color = _ACTIVE_COLOR
            else:
                color = _ID_COLOR
            # write the identity at that location
            cv2.putText(img, str(identity), (int(center.y), int(center.x)),
                        cv2.FONT_HERSHEY_PLAIN, 1.25, color, 2,
                        lineType=cv2.LINE_AA)
