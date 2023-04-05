import feather
import numpy as np
import pandas as pd
from scipy import signal

import warnings
from scipy.stats import kurtosis
from scipy.stats import skew
warnings.filterwarnings("ignore")
import os, sys, argparse

# Read data from feather file and return a dataframe
def read_data(file_path):
	df = feather.read_dataframe(file_path)
	n_row = df.shape[0]
	df = df.dropna()
	n_wake = sum(df["Sleep Stage"] == "Wake")
	n_nrem = sum(df["Sleep Stage"] == "NREM")
	n_rem = sum(df["Sleep Stage"] == "REM")
	df["video"] = df["video"].str.replace("M_3mos_B6-W#", "")
	df["video"] = df["video"].str.replace("PI0", "")
	mapping = {'Wake': 0, 'NREM': 1, 'REM': 2}
	#     mapping = {1: 0, 2: 1,3: 2}
	df["Stage"] = [mapping[item] for item in df["Sleep Stage"]]
	df["wl_ratio"] = df["w"] / df["l"]
	df['dx'] = df["x"].diff()
	df['dy'] = df["y"].diff()
	df["dx2_plus_dy2"] = np.square(df["dx"]) + np.square(df["dy"])
	df = df.dropna()
	return df

# Generate features from each signal per epoch 
def get_freq_feature(wave, a, b, samplerate):
	wave = signal.filtfilt(a, b, wave)
	freqs, psd = signal.welch(wave)
	freqs = freqs * samplerate
	k = kurtosis(wave)
	k_psd = kurtosis(psd)
	s_psd = skew(psd)
	idx_1 = np.logical_and(freqs >= 0.1, freqs < 1)
	idx_3 = np.logical_and(freqs >= 1, freqs < 3)
	idx_5 = np.logical_and(freqs >= 3, freqs < 5)
	idx_8 = np.logical_and(freqs >= 5, freqs < 8)
	idx_15 = np.logical_and(freqs >= 8, freqs < 15)
	MPL_1 = np.mean(psd[idx_1])
	MPL_3 = np.mean(psd[idx_3])
	MPL_5 = np.mean(psd[idx_5])
	MPL_8 = np.mean(psd[idx_8])
	MPL_15 = np.mean(psd[idx_15])
	Tot_PSD = np.sum(psd)
	Max_PSD = max(psd)
	Min_PSD = min(psd)
	Ave_PSD = np.average(psd)
	Std_PSD = np.std(psd)
	Ave_Signal = np.average(wave)
	Std_Signal = np.std(wave)
	Max_Signal = max(wave)
	Min_Signal = min(wave)
	Top_Signal = freqs[psd == Max_PSD][0]
	Med_Signal = np.median(wave)
	Med_PSD = np.median(psd)
	return [k, k_psd, s_psd, MPL_1, MPL_3, MPL_5, MPL_8, MPL_15, Tot_PSD, Max_PSD, Min_PSD, Ave_PSD, Std_PSD,
			Ave_Signal, Std_Signal, Max_Signal, Min_Signal, Top_Signal, Med_Signal, Med_PSD]

# For each signal, run get_freq_feature
def get_features_per_epoch(df_epoch, a, b, samplerate, signals):
	df_epoch = df_epoch.reset_index(drop=True)
	epoch = df_epoch["unique_epoch_id"].unique()[0]
	res = [epoch, df_epoch["video"].unique()[0], df_epoch["Stage"].unique()[0]]
	df_epoch["dx"].iloc[0] = 0
	df_epoch["dy"].iloc[0] = 0
	df_epoch["dx2_plus_dy2"].iloc[0] = 0
	for signal in signals:
		freq_features = get_freq_feature(df_epoch[signal].values, a, b, samplerate)
		res.extend(freq_features)
	return res

# For each epoch, run get_features_per_epoch
def get_features(df):
	signals = ['m00', 'perimeter', 'w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2", 'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6']
	colnames_surfix = ["__k", "__k_psd", "__s_psd", "__MPL_1", "__MPL_3",
					   "__MPL_5", "__MPL_8", "__MPL_15", "__Tot_PSD", "__Max_PSD", "__Min_PSD",
					   "__Ave_PSD", "__Std_PSD", "__Ave_Signal", "__Std_Signal",
					   "__Max_Signal", "__Min_Signal", "__Top_Signal", "__Med_Signal", "__Med_PSD"]
	colnames = ["unique_epoch_id", "video", "Stage"] + [s + c for s in signals for c in colnames_surfix]
	features = ['m00', 'perimeter', 'w', 'l', "wl_ratio", "dx", "dy", "dx2_plus_dy2", 'hu0', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6']
	samplerate = 30.
	a, b = signal.butter(5, [1. / samplerate, 29. / samplerate], 'bandpass')
	df_res = []
	i=0
	for epoch in df["unique_epoch_id"].unique():
		df_epoch = df[df["unique_epoch_id"] == epoch]
		df_res.append(get_features_per_epoch(df_epoch, a, b, samplerate, features))
		# if (i%100==0):
		# 	print(i)
		i=i+1
	df_res = pd.DataFrame(df_res, columns=colnames)
	return df_res

# Split dataframe into smaller chunks to run the program in parallel 
def splitDataFrameIntoSmaller(df, chunkSize = 1000):
	epoches = df["unique_epoch_id"].unique()
	n=len(epoches)
	listOfDf = list()
	numberChunks = n // chunkSize + 1
	for i in range(numberChunks):
		listOfDf.append(df[df["unique_epoch_id"].isin(epoches[i*chunkSize:(i+1)*chunkSize])])
	return listOfDf

# Main function for handling read/writes
def generate_features(args):
	if args.output is None:
		out_fname = os.path.splitext(args.input)[0] + '.csv'
	else:
		out_fname = args.output
	df = read_data(args.input)
	df_count = df.groupby(['unique_epoch_id']).size().reset_index(name='counts')
	# Remove any epochs that are missing too many frames
	epochs = df_count[df_count['counts'] > 80].unique_epoch_id.values
	df = df[df['unique_epoch_id'].isin(epochs)].reset_index(drop=True)
	if args.use_mp is not None:
		# Process the data in chunks
		import multiprocessing as mp
		pool = mp.Pool(args.use_mp)
		chunks = splitDataFrameIntoSmaller(df, 1000)
		result = pool.starmap(get_features, zip(chunks))
		all_features = pd.concat(result)
	else:
		all_features = get_features(df)
	# Write data out
	all_features.to_csv(out_fname)

def main(argv):
	parser = argparse.ArgumentParser(description='Converts a feather segmentation file into a csv feature file')
	parser.add_argument('--input', help='Input feather file', required=True)
	parser.add_argument('--output', help='Output csv file (if other than input.csv)', default=None)
	parser.add_argument('--use_mp', help='Use multiprocessing library for computing features', default=None, type=int)
	args = parser.parse_args()
	generate_features(args)

if __name__ == '__main__':
	main(sys.argv[1:])