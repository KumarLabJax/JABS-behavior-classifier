#Per-Frame Features

## Egocentric Animal Features

112 features from 12 animal keypoints

* point mask (boolean, 12 values)
* pairwise distances between all points (66 values)
* angles between a subset of connected points (11 values)
    * nose, base neck, right front paw
    * nose, base neck, left front paw
    * right front paw, base neck, center spine
    * left front paw, base neck, center spine
    * base neck, center spine, base tail
    * right rear paw, base tail, center spine
    * left rear paw, base tail, center spine
    * right rear paw, base tail, mid tail
    * left rear paw, base tail, mid tail
    * center spine, base tail, mid tail
    * base tail, mid tail, tip tail
* keypoint speeds (12 values)
* angular velocity (using base tail → base neck bearing)
* velocities (each velocity listed below consists of two features – the direction and magnitude components)
    * centroid
    * nose velocity
    * base tail
    * left front paw
    * right front paw

## Social features (v3 pose files)

21 features from social context

* distance to closest mouse
* distance to closest in field of view (fov)
* fov angle
* pairwise distances (9 values)
    * pairwise distances between (nose, base neck, tail) points of subject and closest mouse
* pairwise distances fov (9 values)
    * pairwise distances between (nose, base neck, tail) points of subject and closest fov mouse

## Static object features (v5 pose files)

### Arena Corners

2 features from arena corners

* distance to corner using the convex hull center
* bearing to corner using angle of the base neck - nose vector

### Water Spout (Lixit)

1 feature from lixit

* distance from nose to nearest lixit

### Food Hopper

10 features from food hopper

* signed distance from keypoint to food hopper border (positive = inside, negative = outside)
    * nose
    * left ear
    * right ear
    * base neck
    * left front paw
    * right front paw
    * center spine
    * left rear paw
    * right rear paw
    * base tail

# Window Features

Window features describe how a specific per-frame feature may be changing over time. We use a centered window around the frame of interest. These calculated features are added to the total feature vector. Window features calculated are different whether or not the per-frame feature is non-circular or circular.

## Non circular measurements

* mean
* median
* standard deviation
* max
* min

## Circular measurements

* circstd
* cirmean

### Circular Feature List

* Angles
* Bearings
