import numpy as np
# import sys
# sys.path.append('/media/bgeuther/Storage/TempStorage/JABS-behavior-classifier/')
# from src.project import Project
# from src.classifier import Classifier
import scipy
import h5py
# import plotnine as p9

# Change the test project folder to where the test data
# Sample file used is sample_pose_est_v6.h5.gz inside tests/data/
# For convenience, I extracted it into this folder
project_folder = '/media/bgeuther/Storage/TempStorage/liftover/seg-testing3'
test_vid = 'sample'
# Test animal to read in
animal_idx = 2 # for this test animal, this happens to be the "first"
# Test frame to work with
# This animal is only missing for frames 689 and 690
# The nose keypoint is only missing for frames 1160

# Pick some interesting data location to check calculations
cur_frame = 1700
window_size = 5*5
fps = 30

# Comments for when these features start getting cached
# These 2 cache + feature files are initialized using initialize_project.py
# Since initializeproject needs a video, we generated a blank one:
# ffmpeg -t 120 -s 800x800 -r 30 -f rawvideo -pix_fmt rgb24 -i /dev/zero -c:v libx264 -pix_fmt yuv420p sample.avi

# with h5py.File(project_folder + '/rotta/features/' + test_vid + '/' + str(animal_idx-1) + '/per_frame.h5', 'r') as f:
# 	valid_frame = f['frame_valid'][:]
# 	nose_speed_features = f['features/point_speeds'][:,0]

# This will need to be changed based on where the data actually gets placed...
# with h5py.File(project_folder + '/rotta/features/' + test_vid + '/' + str(animal_idx-1) + '/window_features_5.h5'):
# 	nose_fft_features = f['features/point_speeds/signal?'][:]

# Read in the original data
with h5py.File(project_folder + '/' + test_vid + '_pose_est_v6.h5', 'r') as f:
	raw_pose_data = f['poseest/points'][:]
	raw_embed_data = f['poseest/instance_embed_id'][:]
	cm_per_px2 = f['poseest'].attrs['cm_per_pixel']

# This is done internally when reading in the pose data
# Since there is still that "smoothing" bug, we calculate them manually
points_for_individual = np.zeros(np.delete(np.shape(raw_pose_data),1), dtype=raw_pose_data.dtype)
idxs = np.where(raw_embed_data==animal_idx)
points_for_individual[idxs[0],:,:] = raw_pose_data[idxs[0], idxs[1],:,:]

nose_speeds = np.gradient(points_for_individual[:,0,:], axis=0)
nose_speeds = np.hypot(nose_speeds[:,0], nose_speeds[:,1])*cm_per_px2*fps

# Plot to confirm speed calculation looks correct
#(p9.ggplot(p9.aes(x=np.arange(len(nose_speeds)), y=nose_speeds))+p9.geom_line()+p9.geom_point()+p9.geom_line(p9.aes(x=np.arange(len(nose_speeds)), y=nose_speed_features), color='red')+p9.theme_bw()).draw().show()

signal = nose_speeds[cur_frame-window_size:cur_frame+window_size+1]
# This will be the "correct" version with smoothed base features...
# signal = np.convolve(nose_speeds, np.ones(3)/3, 'same')[cur_frame-window_size:cur_frame+window_size+1]
# or
# signal = nose_speed_features[cur_frame-window_size:cur_frame+window_size+1]

samp_freqs, amp_freqs = scipy.signal.welch(signal, fs=fps, nperseg=16, nfft=64)
# nperseg is the number of samples in the sliding window? typically is a power of 2
# this value will provide a warning if the number of samples in the signal is < this (saying it adjusted it)
# nfft controls the 0-padding window for the fft, since we're looking at tiny windows this will give more bins

# (p9.ggplot(p9.aes(x=samp_freqs, y=amp_freqs))+p9.geom_line()+p9.geom_point()+p9.theme_bw()).draw().show()
# (p9.ggplot(p9.aes(x=np.arange(len(signal))/30, y=signal))+p9.geom_line()+p9.geom_point()+p9.theme_bw()).draw().show()


# Control numbers tested just to confirm 0-padding numbers were fine...
# dummy signal @ a frequency
# signal = amp*np.sin(2*np.pi*time*freq)
# sample a signal with amplitude of 2 with frequency of 5Hz 11 times @ 30Hz
# signal = 2*np.sin(2*np.pi*np.arange(11)/30*5)
# run calculations
# samp_freqs, amp_freqs = scipy.signal.welch(signal, fs=fps, nperseg=16, nfft=64)
# # nfft controls the 0-padding for the fft, since we're looking at tiny windows
# (p9.ggplot(p9.aes(x=samp_freqs, y=amp_freqs))+p9.geom_line()+p9.geom_point()+p9.theme_bw()).draw().show()
# (p9.ggplot(p9.aes(x=np.arange(len(signal))/30, y=signal))+p9.geom_line()+p9.geom_point()+p9.theme_bw()).draw().show()
# Peak is wide, but centered around 5 (the selected frequency) -- check
# Amplitude is much smaller than 2, but that's because we only have a small portion of the wave

# 0-1Hz
psd_1 = np.mean(amp_freqs[np.logical_and(samp_freqs > 0.1, samp_freqs < 1)])
# 1-3Hz
psd_2 = np.mean(amp_freqs[np.logical_and(samp_freqs > 1, samp_freqs < 3)])
# 3-5Hz
psd_3 = np.mean(amp_freqs[np.logical_and(samp_freqs > 3, samp_freqs < 5)])
# 5-8Hz
psd_4 = np.mean(amp_freqs[np.logical_and(samp_freqs > 5, samp_freqs < 8)])
# 8-15Hz
psd_5 = np.mean(amp_freqs[np.logical_and(samp_freqs > 8, samp_freqs < 15)])

psd_tot = np.sum(amp_freqs)
psd_max = np.max(amp_freqs)
psd_min = np.min(amp_freqs)
psd_avg = np.average(amp_freqs)
psd_med = np.median(amp_freqs)
psd_std = np.std(amp_freqs)
psd_kur = scipy.stats.kurtosis(amp_freqs)
psd_ske = scipy.stats.skew(amp_freqs)

top_freq = samp_freqs[amp_freqs == psd_max][0]

[psd_1, psd_2, psd_3, psd_4, psd_5, psd_tot, psd_max, psd_min, psd_avg, psd_med, psd_std, psd_kur, psd_kur, psd_ske, top_freq]
# [1.0497800464150113, 2.711189963463685, 4.614291627550688, 4.52877200769071, 7.130551751551292, 165.07354943080787, 10.929267288951255, 0.2630445001228418, 5.002228770630541, 4.672686132477225, 2.6563094001284804, -0.04479529828345585, -0.04479529828345585, 0.6045623152828209, 10.78125]
