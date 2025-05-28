from utils import * 
import sys
import yaml
import os
from scipy.io import wavfile
from noise_reduction import *
import pandas as pd
import time
import multiprocessing

# config_file = sys.argv[1]
# #Defining parameters
# with open(config_file) as f:
#	 cfg = yaml.load(f, Loader=yaml.FullLoader)

# audio_file_type = cfg['audio']['file_type']

def modify_matrix(matrix, start_pixel_time, end_pixel_time, start_pixel_freq, end_pixel_freq, X):
    """
    Changes all values of a 2D matrix to X except for the specified interval.

    Parameters:
    - matrix (2D numpy array): The input matrix to modify.
    - start_pixel_time (int): The start index for the time dimension.
    - end_pixel_time (int): The end index for the time dimension.
    - start_pixel_freq (int): The start index for the frequency dimension.
    - end_pixel_freq (int): The end index for the frequency dimension.
    - X (any): The value to set for the matrix values outside the specified interval.

    Returns:
    - Modified matrix with all values set to X except the specified interval.
    """
    # Create a copy of the matrix to avoid modifying the original matrix
    modified_matrix = np.copy(matrix)
    
    # Set the values to X outside the specified interval
    modified_matrix[:,:start_pixel_time] = modified_matrix[:,:start_pixel_time] * X
    modified_matrix[:,end_pixel_time:] = modified_matrix[:,end_pixel_time:] * X
    modified_matrix[:start_pixel_freq, start_pixel_time:end_pixel_time] = modified_matrix[:start_pixel_freq, start_pixel_time:end_pixel_time] * X
    modified_matrix[end_pixel_freq:, start_pixel_time:end_pixel_time] = modified_matrix[end_pixel_freq:, start_pixel_time:end_pixel_time] * X
    
    return modified_matrix

def crop_spectrogram(spec_data, spec_time_duration, spec_freq_interval, time_start_interval, time_end_interval, freq_start_interval, freq_end_interval, window = False, force_min = True):
	"""
	Crop the spectrogram image based on the time and frequency intervals.
	
	Parameters:
	- image_path (str): Path to the input spectrogram image.
	- spec_time_duration (float): Total time duration of the spectrogram in seconds (audio length).
	- spec_freq_interval (float): Total frequency range of the spectrogram in Hz (max frequency).
	- time_start_interval (float): Start time in seconds for cropping.
	- time_end_interval (float): End time in seconds for cropping.
	- freq_start_interval (float): Start frequency in Hz for cropping.
	- freq_end_interval (float): End frequency in Hz for cropping.
	
	Returns:
	- None: Saves the cropped image at the output path.
	"""
	# Get the spectrogram dimensions (time steps and frequency bins)
	num_freq_bins, num_time_steps = spec_data.shape
	print("-----------------------------")
	print('size:', num_time_steps, num_freq_bins)
	
	# Calculate the time and frequency intervals (per pixel)
	time_interval = spec_time_duration / num_time_steps
	freq_interval = spec_freq_interval / num_freq_bins
	# print('interval: ', time_interval, freq_interval)
	# Calculate the pixel positions for cropping based on the provided time and frequency intervals
	start_pixel_time = int(time_start_interval / time_interval)
	end_pixel_time = int(time_end_interval / time_interval) + 1
	start_pixel_freq = int(freq_start_interval / freq_interval)
	end_pixel_freq = int(freq_end_interval / freq_interval) + 1
	# print('time_splits: ', start_pixel_time, end_pixel_time)
	# print('freq_splits: ', start_pixel_freq, end_pixel_freq)
	 # Ensure the cropping positions are within the bounds of the array
	start_pixel_time = max(start_pixel_time, 0)
	end_pixel_time = min(end_pixel_time, num_time_steps)
	start_pixel_freq = max(start_pixel_freq, 0)
	end_pixel_freq = min(end_pixel_freq, num_freq_bins)
	
	# Crop the data using numpy slicing (Time -> rows, Frequency -> columns)
	# cropped_spec_data = spec_data[num_freq_bins - end_pixel_freq: num_freq_bins - start_pixel_freq, start_pixel_time:end_pixel_time]
	time_interval_samples = end_pixel_time - start_pixel_time
	freq_interval_samples = end_pixel_freq - start_pixel_freq
	print(time_interval_samples, freq_interval_samples)
	if(window == True):
		cropped_spec_data = modify_matrix(spec_data, start_pixel_time, end_pixel_time, start_pixel_freq, end_pixel_freq, 0.3)
	elif(force_min == True):
		if(time_interval_samples < 128):
			gain_time_pixels = int((128 - time_interval_samples)/2)
			start_pixel_time -= gain_time_pixels
			end_pixel_time += gain_time_pixels
			print('new time pixel interval = ', end_pixel_time - start_pixel_time)
			print(start_pixel_time, end_pixel_time)
			if(start_pixel_time < 0):
				end_pixel_time = 128
				start_pixel_time = 0
			if(end_pixel_time > num_time_steps):
				end_pixel_time = num_time_steps
				start_pixel_time = num_time_steps - 128
		if(freq_interval_samples < 128):	
			gain_freq_pixels = int((128 - freq_interval_samples)/2)
			start_pixel_freq -= gain_freq_pixels
			end_pixel_freq += gain_freq_pixels
			print('new freq pixel interval = ', end_pixel_freq - start_pixel_freq)
			print(start_pixel_freq, end_pixel_freq)
			if(start_pixel_freq < 0):
				end_pixel_freq = 128
				start_pixel_freq = 0
			if(end_pixel_freq > num_freq_bins):
				end_pixel_freq = num_freq_bins
				start_pixel_freq = num_freq_bins - 128
		cropped_spec_data = spec_data[start_pixel_freq:end_pixel_freq, start_pixel_time:end_pixel_time]
	else:
		cropped_spec_data = spec_data[start_pixel_freq:end_pixel_freq, start_pixel_time:end_pixel_time]
	# cropped_spec_data = spec_data[:, start_pixel_time:end_pixel_time]
	return cropped_spec_data

def split_audio_time(signal, sr, start_time, end_time):
    """
    Splits a signal array into a time interval.

    Parameters:
    - signal (np.ndarray): The input signal (audio samples).
    - sr (int): The sample rate of the signal (samples per second).
    - start_time (float): The start time in seconds of the interval.
    - end_time (float): The end time in seconds of the interval.

    Returns:
    - np.ndarray: The signal segment corresponding to the specified time interval.
    """
    # Calculate the start and end indices based on the sample rate
    start_sample = max(int(start_time * sr), 0) # Convert start time to sample index
    end_sample = min(int(end_time * sr), len(signal))      # Convert end time to sample index

    # Check if indices are within the bounds of the signal
    if start_sample < 0 or end_sample > len(signal):
        raise ValueError("Time interval exceeds signal length")

    # Extract the segment from the signal array
    return signal[start_sample:end_sample]

def split_crop_spec(filepath_spec, img_start_time, img_end_time):
	sig, sr = load_signal(filepath_audio)
	return sig[int(start_time * sr): int(end_time * sr)]

# def split_image_freq(filepath_img, start_freq, end_freq, sr):

def remove_noise_audio(filepath_audio, technique):
	sig, sr = load_signal(filepath_audio)
	# print('lensig: ', len(sig))
	# print('sig: ', sig)
	# print(technique)
	# print('treated:')
	# print(apply_technique_NR(sig, sr, technique))
	return apply_technique_NR(sig, sr, technique), sr

def create_treated_file(filepath_in, filename, filepath_out, technique, split_info, is_signal = False, save_audio = True, save_img = True, split_vocalization = True, split_window = True):
	
	# print('tech_create_file: ', technique)
	sig, sr = load_signal(filepath_in)
	# print(sr)
	treated_data, _ = remove_noise_audio(filepath_in, technique)
	# print(sig)
	# print(sr)
	# print(treated_data)
	# print('file: ', filename)
	# print('len treated sig: ', len(treated_data), ' x ', len(treated_data[0]))
	if(is_signal):
		if(save_audio == True):
			print('############')
			print(treated_data)
			print(sr)
			print(filepath_out)
			print(filename)
			print('############')
			save_signal(treated_data, sr, os.path.join(filepath_out, 'signal', filename))

			if(split_vocalization == True): #CORRIGIR ERRO USANDO 2 VARIAVEIS DE SPLITS: PRA AUDIO E PRA SPEC
				if(filename.split('.')[0] != split_info[0]):
					print("ERROR! Wrong split information passed!")
					print(filename.split('.')[0])
					print(split_info[0])
				new_sig = split_audio_time(treated_data, sr, split_info[1], split_info[2])

				save_signal(new_sig, sr, os.path.join(filepath_out, 'signal_cropped_vocal', filename))

		if(save_img == True):
			spec = signal_to_spec(treated_data, sr)
			save_spec(spec, os.path.join(filepath_out, 'imgs', filename.split('.')[0]))
		if(split_vocalization == True):
			if(filename.split('.')[0] != split_info[0]):
				print("ERROR! Wrong split information passed!")
				print(filename.split('.')[0])
				print(split_info[0])
			else:
				spec = signal_to_spec(treated_data, sr)
				cropped_spec = crop_spectrogram(spec, spec_time_duration=split_info[5],  # Total time of the audio (e.g., 10 seconds)
				spec_freq_interval=sr/2,  # Total frequency range (e.g., 2000 Hz)
				time_start_interval=split_info[1],  # Start time of cropping in seconds
				time_end_interval=split_info[2],  # End time of cropping in seconds
				freq_start_interval=split_info[3],  # Start frequency of cropping in Hz
				freq_end_interval=split_info[4], window = False)
				save_spec(cropped_spec, os.path.join(filepath_out, 'imgs_cropped_min', filename.split('.')[0]))
		if(split_window == True):
			if(filename.split('.')[0] != split_info[0]):
				print("ERROR! Wrong split information passed!")
				print(filename.split('.')[0])
				print(split_info[0])
			else:
				spec = signal_to_spec(treated_data, sr)
				cropped_spec = crop_spectrogram(spec, spec_time_duration=split_info[5],  # Total time of the audio (e.g., 10 seconds)
				spec_freq_interval=sr/2,  # Total frequency range (e.g., 2000 Hz)
				time_start_interval=split_info[1],  # Start time of cropping in seconds
				time_end_interval=split_info[2],  # End time of cropping in seconds
				freq_start_interval=split_info[3],  # Start frequency of cropping in Hz
				freq_end_interval=split_info[4], window = True)
				save_spec(cropped_spec, os.path.join(filepath_out, 'imgs_window', filename.split('.')[0]))
	else:
		print('############')
		print(treated_data)
		print(sr)
		print(filepath_out)
		print(filename)
		print('############')
		if(save_audio == True):
			sig = spec_to_signal(treated_data)
			save_signal(sig, sr, os.path.join(filepath_out, 'signal', filename))

			if(filename.split('.')[0] != split_info[0]):
				print("ERROR! Wrong split information passed!")
			new_sig = split_audio_time(sig, sr, split_info[1], split_info[2])
			save_signal(new_sig, sr, os.path.join(filepath_out, 'signal_cropped_vocal', filename))
		
		if(save_img == True):
			save_spec(treated_data, os.path.join(filepath_out, 'imgs', filename.split('.')[0]))
		if(split_vocalization == True):
			if(filename.split('.')[0] != split_info[0]):
				print("ERROR! Wrong split information passed!")
			else:
				cropped_spec = crop_spectrogram(treated_data, spec_time_duration=split_info[5],  # Total time of the audio (e.g., 10 seconds)
				spec_freq_interval=sr/2,  # Total frequency range (e.g., 2000 Hz)
				time_start_interval=split_info[1],  # Start time of cropping in seconds
				time_end_interval=split_info[2],  # End time of cropping in seconds
				freq_start_interval=split_info[3],  # Start frequency of cropping in Hz
				freq_end_interval=split_info[4], window = False)
				save_spec(cropped_spec, os.path.join(filepath_out, 'imgs_cropped_min', filename.split('.')[0]))
		if(split_window == True):
			if(split_vocalization == True):
				if(filename.split('.')[0] != split_info[0]):
					print("ERROR! Wrong split information passed!")
				else:
					cropped_spec = crop_spectrogram(treated_data, spec_time_duration=split_info[5],  # Total time of the audio (e.g., 10 seconds)
					spec_freq_interval=sr/2,  # Total frequency range (e.g., 2000 Hz)
					time_start_interval=split_info[1],  # Start time of cropping in seconds
					time_end_interval=split_info[2],  # End time of cropping in seconds
					freq_start_interval=split_info[3],  # Start frequency of cropping in Hz
					freq_end_interval=split_info[4], window = True)
					save_spec(cropped_spec, os.path.join(filepath_out, 'imgs_window', filename.split('.')[0]))

def create_treated_files_batch(filepath_in, filepath_out, technique, config, doParallel=True):
	filepaths_audio, filename = get_files_and_paths(filepath_in)
	sorted_files = sorted(zip(filepaths_audio, filename), key=lambda x: x[1])
	# print(sorted_files)
	filepaths_audio, filename = zip(*sorted_files)

	create_folder(filepath_out)
	filepath_out = os.path.join(filepath_out, technique)
	create_folder(filepath_out)
	create_folder(os.path.join(filepath_out, 'signal'))
	create_folder(os.path.join(filepath_out, 'signal_cropped_vocal'))
	create_folder(os.path.join(filepath_out, 'imgs'))
	# create_folder(os.path.join(filepath_out, 'imgs_cropped_vocal'))
	create_folder(os.path.join(filepath_out, 'imgs_window'))
	create_folder(os.path.join(filepath_out, 'imgs_cropped_min'))

	if(config['pipeline']['transform']['save_split_spec_vocalization'] == True or config['pipeline']['transform']['save_window_spec_vocalization'] == True ):
		filepath_label = os.path.join(config['filepath_outputs']['label'], 'labels_LEEC.csv')
		print(filepath_label)
		df = pd.read_csv(filepath_label)
		print(df)
		df = df[df['Filepath_audio'] != 'File not found!']
		print(df)
		df = df[['Final_filename', 'Vocal_start_time', 'Vocal_end_time', 'Low_freq', 'High_freq']] 
		df['Audio_duration'] = config['audio_split']['split_interval']
		df = df.sort_values(by='Final_filename')
		split_info = df.values.tolist()
		print(split_info)
		# print(len(filepaths_audio))
		# print(len(filename))
		# print(filename)
		split_spec_vocalization = False
		split_window_vocalization = False

		if(config['pipeline']['transform']['save_split_spec_vocalization'] == True):
			split_spec_vocalization = True
		if(config['pipeline']['transform']['save_window_spec_vocalization'] == True):
			split_window_vocalization = True
		print("oi")
	else:
		split_info = [[] for _ in range(len(filename))]
		split_spec_vocalization = False
		split_window_vocalization = False
		# print(split_info)

	is_signal = config['techniques'][technique]['output_type']
	if(is_signal == 'signal'):
		is_signal = True
	else:
		is_signal = False

	save_audio = config['pipeline']['transform']['save_transformed_audio']
	save_img = config['pipeline']['transform']['save_transformed_image']
	print("oi2")
	if doParallel:
		start_time = time.time()
		cores = min(multiprocessing.cpu_count(), 1)
		# print('CPU: ', multiprocessing.cpu_count())
		pool = multiprocessing.Pool(cores)
		results = pool.starmap(create_treated_file, 
							 [(filepath, fname, filepath_out, technique, split_info, is_signal, save_audio, save_img, split_spec_vocalization, split_window_vocalization) for filepath, fname, split_info in zip(filepaths_audio, filename, split_info)])
		print("--- %s seconds ---" % (time.time() - start_time))
	else:
		start_time = time.time()
		for i, filepath in enumerate(filepaths_audio):
			print(filename[i])
			print(split_info)
			create_treated_file(filepath, filename[i], filepath_out, technique, split_info[i], is_signal, save_audio, save_img, split_spec_vocalization, split_window_vocalization)
		print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
	filepath_in = 'C:\\Users\\GustavoLopes\\Documents\\Dados\\teste\\Gerados\\splits_s_5'
	filepath_out = 'C:\\Users\\GustavoLopes\\Documents\\Dados\\teste\\Gerados\\transformed'
	technique = 'towsey'
	create_treated_files_batch(filepath_in, filepath_out, technique, is_signal = False, doParallel = True)

'''
if(cfg['pipeline']['split_audio'] == True):
	print('Splitting audio')
	cut_time, cut_type = cfg['audio_split']['split_interval'], cfg['audio_split']['split_type']

	filepath_out = cfg['filepath_outputs']['audio_splits']
	create_folder(filepath_out)

	filepath_label = os.path.join(cfg['filepath_outputs']['label'], 'labels.csv')
	df = pd.read_csv(os.path.join(cfg['filepath_outputs']['label'], 'labels.csv'), sep= '\t') 
	# print(df)
	files = list(set(df['Filename_audio']))
	# print(files)
	for file in files:
		lines = (df[df['Filename_audio'] == file])#.values.tolist()
		# print(lines)
		# print("#########")
		filepath_audio = lines.iloc[0]['Filepath_audio']
		if(filepath_audio == 'File not found!'):
			print("File not found!")
			continue
		sig, sr = load_signal(filepath_audio)
		for _, line in lines.iterrows():
			# print(line)
			start, end, selection = line['Start_time'], line['End_time'], line['Selection']
			# print(start, end, selection)

			splits = audio_splitter(sig, cut_type, cut_time, start, end)
			for i, split in enumerate(splits):
				new_sig = sig[int(split[0] * sr): int(split[1] * sr)]

				save_signal(new_sig, sr, os.path.join(filepath_out, file.split('.')[0] + '_selec_' + str(selection) + '_split_' + str(i) + '.wav'))


transform = cfg['pipeline']['transform']

if(transform['do'] == True):
	print("Transforming data")

	filepath_audios = cfg['filepath_inputs']['audio_transformed']
	# print('files: ', filepath_audios)
	filepath_audio, files = get_files_and_paths(filepath_audios)


	techniques = cfg['techniques']['selected']
	for technique in techniques:
		tech_out_type = cfg['techniques'][technique]['output_type']
		print("Using", technique, "technique")

		filepath_out = cfg['filepath_outputs']['audio_transformed']
		create_folder(filepath_out)
		filepath_out = os.path.join(filepath_out, technique)
		create_folder(filepath_out)
		create_folder(os.path.join(filepath_out, 'signal'))
		create_folder(os.path.join(filepath_out, 'imgs'))

		if(tech_out_type == 'signal'):
			# print('entrou signal')
			# print(filepath_audio)

			for i, file in enumerate(filepath_audio):
				# print("#################")
				# print(file)
				sig, sr = load_signal(file)
				# print(sig)
				new_sig = apply_technique_NR(sig, sr, technique)
				# print("new_sig: ", new_sig)
				if(transform['save_transformed_audio'] == True):
					save_signal(new_sig, sr, os.path.join(filepath_out, 'signal', files[i]))
				if(transform['save_transformed_image'] == True):
					spec = signal_to_spec(new_sig, sr)
					save_spec(spec, os.path.join(filepath_out, 'imgs', files[i].split('.')[0]))

		if(tech_out_type == 'spec'):
			# print('entrou spec')
			# print(filepath_audio)
			for i, file in enumerate(filepath_audio):
				print("#################")
				print(file)
				sig, sr = load_signal(file)
				print("sig: ", sig)
				spec = apply_technique_NR(sig, sr, technique)
				print(spec)
				if(transform['save_transformed_audio'] == True):
					new_sig = spec_to_signal(spec)
					save_signal(new_sig, sr, os.path.join(filepath_out, 'signal', files[i]))
				if(transform['save_transformed_image'] == True):
					save_spec(spec, os.path.join(filepath_out, 'imgs', files[i].split('.')[0]))


'''