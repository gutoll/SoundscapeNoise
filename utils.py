import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2

def get_files_and_paths(filepath, file_type = 'wav'):
	list_filepaths = []
	list_files = []
	# print('entrou')
	for path, dirs, files in os.walk(filepath):
		# print("arquivos: ", files)
		for file in files:
			# print('achou arquivo: ', file)
			if (file.endswith(file_type)):
				list_filepaths.append(os.path.join(path, file))
				list_files.append(file)
	# print('#############')
	# print(list_files[0:3])
	return list_filepaths, list_files

def has_files(direc):
	for path, dirs, files in os.walk(filepath):
		if files:
			return True
		else:
			return False

def signal_to_spec(sig, sr):

	# S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=128, fmax=8000)
	# S_dB = librosa.power_to_db(S, ref=np.max) + 96

	D = librosa.stft(sig)
	# print(len(D), len(D[0]))
	# Convert to magnitude spectrogram
	S, phase = librosa.magphase(D)
	# print(len(S), len(S[0]))
	S_dB = librosa.amplitude_to_db(S, ref=np.max) + 96
	# print(len(S_dB), len(S_dB[0]))
	return S_dB


def create_folder(filepath):
	filepath = Path(filepath)
	if(not filepath.exists()):
		os.makedirs(filepath)


def spec_to_signal(spec):
	# print("##################")
	# print(spec)
	# print(np.isnan(spec).any())
	sig = librosa.feature.inverse.mel_to_audio(spec)
	return sig

def load_signal(filename):
	audio, sr = sf.read(filename)
	return audio, sr

def save_signal(sig, sr, filename):
	# print(sig)
	# breakpoint()
	sf.write(filename, sig, sr)


def save_spec(spec, filename, size_spec = (256, 256)):
	# image = cv2.imdecode(spec, 1)
	spec = cv2.resize(spec, size_spec,interpolation=cv2.INTER_AREA)
	fig = plt.figure(figsize=size_spec, frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	# ax.invert_yaxis()
	fig.add_axes(ax)
	plt.imshow(spec)
	plt.gcf().set_size_inches(4, 4)
	plt.gca().invert_yaxis()
	plt.savefig(filename + '.png', dpi=64)
	plt.close()

def label_audio_splitter(cut_type, cut_time, start_split, end_split, file_duration):
	splits = []
	vocal_duration = end_split - start_split
	if(cut_type == 'c'): #centralized
		central_time = (vocal_duration/2) + start_split
		start_time = central_time - (cut_time / 2)
		if(start_time < 0):
			start_time = 0
			end_time = cut_time
		end_time = central_time + (cut_time / 2)
		if(end_time > file_duration):
			end_time = file_duration
			start_time = file_duration - cut_time
		# new_start_time = 
		# new_end_time = end_time - new_start_time
		if(start_split - start_time < 0):
			new_start_time = 0
			new_end_time = new_start_time + cut_time
		else:
			new_start_time = start_split - start_time
			if(vocal_duration > 0.5):
				new_end_time = new_start_time + vocal_duration
			else:
				new_end_time = new_start_time + 0.5
		splits.append([start_time, end_time, new_start_time, new_end_time])
	elif(cut_type == 's'): #splits
		num_splits = int(((end_split - start_split) / cut_time) // 1)	#rounding down the number of times cut_time fits inside vocalization label
																#last split will be done backwards
		for i in range(num_splits):
			start_time = start_split + i * cut_time
			end_time = start_split + (i + 1) * cut_time
			splits.append([start_time, end_time, 0, cut_time])
		if(end_split - cut_time > 0):
			if(num_splits > 0):
				splits.append([end_split - cut_time, end_split, 0, cut_time])
			else:
				if(vocal_duration > 0.5):
					splits.append([end_split - cut_time, end_split, cut_time - (vocal_duration), cut_time])
				else:
					splits.append([end_split - cut_time, end_split, cut_time - 0.5, cut_time])
		else:
			if(vocal_duration > 0.5):
				splits.append([start_split, start_split + cut_time, 0, end_split - start_split])
			else:
				splits.append([start_split, start_split + cut_time, 0, 0.5])
	elif(cut_type == 'v'): # full vocalization
		if(vocal_duration > 0.5):
			splits.append([start_split, end_split, 0, end_split - start_split])
		else:
			if(start_split + 0.5 < file_duration):
				splits.append([start_split, start_split + 0.5, 0, 0.5])
			else:
				splits.append([end_split - 0.5, end_split, 0, 0.5])
	return splits