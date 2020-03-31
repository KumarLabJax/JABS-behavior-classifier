import cv2
from shapely.geometry import MultiPoint
from itertools import compress


def annotate_image(img, points, mask):
    if points is not None:
        center = get_centroid(points, mask)
        cv2.circle(img, (int(center.y), int(center.x)), 5, (255, 0, 0), -1)


def get_centroid(points, mask):
    filtered_points = list(compress(points[:-2], mask[:-2]))
    shape = MultiPoint(filtered_points).convex_hull
    return shape.centroid
