import subprocess
from utils import * 
import sys
import yaml
import os
import pandas as pd
from compile_labels import generate_label_doc
from transform_data import split_audio_time, create_treated_files_batch
import utils
from pydub import AudioSegment
import settings
from AudioTools.indices import AcousticIndices
import rpy2.robjects as ro

import warnings

# Suppress the CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Blowfish.*")



if __name__ == '__main__':
	config_file = sys.argv[1]

	#Defining parameters

	with open(config_file) as f:
	    cfg = yaml.load(f, Loader=yaml.FullLoader)


	pipeline = cfg['pipeline']

	audio_file_format = cfg['audio']['file_type']
	if(audio_file_format != 'wav'):
		print("Converting audio type")
		subprocess.call(['python', 'convert.py', config_file])

	if(pipeline['compile_labels'] == True):
		print("Compiling labels")
		# subprocess.call(['python', 'compile_labels.py', config_file])
		generate_label_doc(cfg, 'labels_LEEC.csv')

	if(pipeline['split_audio'] == True):
		print("Spliting audio")
		filepath_label = os.path.join(cfg['filepath_outputs']['label'], 'labels_LEEC.csv')
		filepath_output = os.path.join(cfg['filepath_outputs']['audio_splits'])
		folder_name_splits = os.path.join(filepath_output, 'splits_' + cfg['audio_split']['split_type'] + '_' + str(cfg['audio_split']['split_interval']))
		create_folder(folder_name_splits)

		df = pd.read_csv(filepath_label)

		for i, row in df.iterrows():
			if(row['Filepath_audio'] != 'File not found!'):
				filepath_audio = row['Filepath_audio']
				start_time = row['Split_start_time']
				end_time = row['Split_end_time']
				filename_output =  row['Final_filename']
				sig, sr = load_signal(filepath_audio)
				new_audio = split_audio_time(sig, sr, start_time, end_time)
				save_signal(new_audio, sr, os.path.join(folder_name_splits, filename_output + '.wav'))

	if(pipeline['transform']['do'] == True):
		print("Transforming data")
		techniques = cfg['techniques']['selected']
		filepath_in = os.path.join(cfg['filepath_inputs']['audio_transformed'], 'splits_' + cfg['audio_split']['split_type'] + '_' + str(cfg['audio_split']['split_interval']))
		filepath_out = cfg['filepath_outputs']['audio_transformed']
		split_spec = cfg['pipeline']['transform']['save_split_spec_vocalization']
		for technique in techniques:
			print('Removing noise using technique: ', technique)
			
			create_treated_files_batch(filepath_in, filepath_out, technique, config = cfg, doParallel = False)
		# subprocess.call(['python', 'transform_data.py', config_file])

	if(pipeline['generate_indices'] == True):
		techniques = cfg['techniques']['selected']
		os.chdir("./AudioTools")
		for technique in techniques:
			folder = os.path.join(cfg['filepath_outputs']['audio_transformed'], technique)#, 'signal_cropped_vocal')
			# print(folder)
			# print("Current working directory:", os.getcwd())

			a = AcousticIndices()
			print('Generate acoustic indices for technique: ', technique)
			a.process_dir(os.path.join(folder, 'signal'))
			a.process_dir(os.path.join(folder, 'signal_cropped_vocal'))

	# if(pipeline['organize_data'] == True):
	# 	print("Organizing data")
	# 	subprocess.call(['python', 'organize_data.py', config_file])
