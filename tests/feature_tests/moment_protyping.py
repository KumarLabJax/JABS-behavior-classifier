import sys
import numpy as np
from itertools import chain
import pandas as pd
from src.project import Project
from src.classifier import Classifier
from src.feature_extraction.base_features import moments
import sklearn
import h5py
import cv2

# Change the test project folder to where the test data
project_folder = '/media/bgeuther/Storage/TempStorage/liftover/seg-testing3'
test_vid = 'sample'
# Test animal to read in
animal_idx = 3
# Test frame to work with
cur_frame = 1

# These 2 cache + feature files are initialized using initialize_project.py
# Since initializeproject needs a video, we generated a blank one:
# ffmpeg -t 120 -s 800x800 -r 30 -f rawvideo -pix_fmt rgb24 -i /dev/zero -c:v libx264 -pix_fmt yuv420p sample.avi
# 
# python3 JABS-behavior-classifier/initialize_project.py [path_to_project_folder]
with h5py.File(project_folder + '/rotta/cache/' + test_vid + '_pose_est_v6_cache.h5','r') as f:
	poses = f['poseest/points'][:][animal_idx]
	point_mask = f['poseest/point_mask'][:][animal_idx]
	id_mask = f['poseest/identity_mask'][:][animal_idx]
	cm_per_px = f['poseest'].attrs['cm_per_pixel']

with h5py.File(project_folder + '/rotta/features/' + test_vid + '/' + str(animal_idx) + '/per_frame.h5', 'r') as f:
	valid_frame = f['frame_valid'][:]
	jabs_moments = f['features/moments'][:]

# Read in the original data
with h5py.File(project_folder + '/' + test_vid + '_pose_est_v6.h5', 'r') as f:
	seg_data = f['poseest/seg_data'][:]
	seg_external_flag = f['poseest/seg_external_flag'][:]
	track_sorting = f['poseest/instance_track_id'][:]
	embed_sorting = f['poseest/instance_embed_id'][:]
	seg_sorting = f['poseest/instance_seg_id'][:]
	cm_per_px2 = f['poseest'].attrs['cm_per_pixel']

# Trims a single contour
# Returns an opencv-complaint contour (dtype = int)
def trim_seg(arr):
	assert arr.ndim == 2
	return_arr = arr[np.all(arr!=-1, axis=1),:]
	if len(return_arr)>0:
		return return_arr.astype(int)

# Trims all contours for an individual
def trim_seg_list(arr):
	assert arr.ndim == 3
	return [trim_seg(x) for x in arr if np.any(x!=-1)]

# Calculates the moments given a contour list
# Helper for just rendering the contours before calling moments
def get_moments_from_list(contour_list, frame_size=800):
	# Safety if the contours are outside the frame
	max_pos = np.max(np.concatenate(contour_list))
	if max_pos>frame_size:
		frame_size = max_pos+1
	# Render the contours on a frame
	render = np.zeros([frame_size, frame_size, 1], dtype=np.uint8)
	_ = cv2.drawContours(render, contour_list, -1, [1], -1)
	return cv2.moments(render)

# This is the number we observe
jabs_moments[cur_frame][0]
# 3451.0

trimmed_contour_lists = [trim_seg_list(seg_data[cur_frame,x]) for x in np.arange(np.shape(seg_data)[1])]
contour_areas = [[cv2.moments(x)['m00']*(cm_per_px**2) for x in y] for y in trimmed_contour_lists]
# [[27.15308851474372], [27.45793221681897], [27.159373951899912], [0.3708407922152499, 14.673353041127134], []]

# Since it may be difficult to merge/keep track if internal/external, we can just be lazy and render the moments and lean on opencv:
[get_moments_from_list(x)['m00']*cm_per_px**2 for x in trimmed_contour_lists if len(x)>0]
# [28.52331381479329, 28.818729361134253, 28.447888568919, 15.99015212534908]

# Ugly format of nested for loops to calculate things manually
# Note that this takes into account the seg_external_flag for internal/external
[[cv2.moments(trim_seg(np.reshape(seg_data[cur_frame,x,y], [-1,2])))['m00']*(cm_per_px**2)*[-1,1][seg_external_flag[cur_frame,x,y].astype(int)] for y in np.arange(np.shape(seg_data)[2])] for x in np.arange(np.shape(seg_data)[1])]
# [[27.15308851474372, -0.0, -0.0, -0.0], [27.45793221681897, -0.0, -0.0, -0.0], [27.159373951899912, -0.0, -0.0, -0.0], [0.3708407922152499, 14.673353041127134, -0.0, -0.0], [-0.0, -0.0, -0.0, -0.0]]

# Erroroneous method of just flattening the segmentation points all into one
[cv2.moments(trim_seg(np.reshape(seg_data[cur_frame,x], [-1,2]).astype(int)))['m00'] for x in np.arange(np.shape(seg_data)[1])]
# [4320.0, 4368.5, 4321.0, 3451.0, 0.0]

color_set = [
	(28,26,228),		# Red
	(184,126,55),		# Blue
	(0,127,255),		# Orange
	(74,175,75),		# Green
]
# Plot the segmentations
# If you have the video, can render it on top of the video, otherwise just on a black frame...
# import imageio
# vid_reader = imageio.get_reader(project_folder + '/' + test_vid + '.avi')
# frame = vid_reader.get_data(cur_frame)
frame = np.zeros([800,800,3], dtype=np.uint8)
for i,trimmed_contours in enumerate(trimmed_contour_lists):
	if len(trimmed_contours)>0:
		_ = cv2.drawContours(frame, trimmed_contours, -1, color_set[i], -1)

# Plot the segmentations
cv2.imshow('test',frame)
cv2.waitKey()
cv2.destroyAllWindows()
