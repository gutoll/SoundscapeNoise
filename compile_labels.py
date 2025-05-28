import os
import re
import pandas as pd
from utils import * 
import sys
import yaml


# aggregate all information from all label files (for desired species only) and correlates with audio file location
def get_info_labels(filepaths_label, files_label, filepaths_audio, files_audio, config):
	labels = []
	counter = 0
	for file_label_id, filepath_label in enumerate(filepaths_label):
		with open(filepath_label, 'r') as f:
			text = f.read()
			# print(text)
			lines = text.split('\n')
			if(config['labels']['header'] == True):
				lines = lines[1:]
			for i, line in enumerate(lines):
				if(line == ''):
					break
				if(config['labels']['file_sep'] == 'comma'):
					info = line.split(',')
				elif(config['labels']['file_sep'] == 'tab'):
					info = line.split('\t')
				# print(info)
				if (info[config['labels']['pos_columns']['species']] in config['labels']['species']):
					if(config['labels']['pos_columns']['filename'] == 'None'):
						# print('ENTROU1')
						filename_audio = files_label[file_label_id].split('.')[0]
					else:
						# print('ENTROU2')
						filename_audio = info[config['labels']['pos_columns']['filename']].split('.')[0]
					if(config['labels']['pos_columns']['selection'] == 'None'):
						selection = str(i)
					else:
						selection = info[config['labels']['pos_columns']['selection']]
					# print(filename_audio)
					if((info[config['labels']['pos_columns']['end_freq']] != info[config['labels']['pos_columns']['start_freq']]) and (info[config['labels']['pos_columns']['end_time']] != info[config['labels']['pos_columns']['start_time']])):
						try:
							file_audio_id = files_audio.index(filename_audio + '.' + config['audio']['file_type'])
							labels.append(filepath_label + '\t' + filepaths_audio[file_audio_id] + '\t' + filename_audio + '\t' + config['audio']['file_type'] + '\t' + selection + '\t' + info[config['labels']['pos_columns']['start_time']] + '\t' + info[config['labels']['pos_columns']['end_time']] + '\t' + info[config['labels']['pos_columns']['start_freq']] + '\t' + info[config['labels']['pos_columns']['end_freq']] + '\t' + info[config['labels']['pos_columns']['species']] + '\t' + str(counter))
						except ValueError:
							labels.append(filepath_label + '\t' + 'File not found!' + '\t' + filename_audio + '\t' + config['audio']['file_type'] + '\t' + selection + '\t' + info[config['labels']['pos_columns']['start_time']] + '\t' + info[config['labels']['pos_columns']['end_time']] + '\t' + info[config['labels']['pos_columns']['start_freq']] + '\t' + info[config['labels']['pos_columns']['end_freq']] + '\t' + info[config['labels']['pos_columns']['species']] + '\t' + str(counter))
						counter += 1
					else:
						print('Vocalization from file', filename_audio, 'with time or frequency range equals to 0')
						try:
							file_audio_id = files_audio.index(filename_audio + '.' + config['audio']['file_type'])
							os.remove(filepaths_audio[file_audio_id])
							print(filepaths_audio[file_audio_id], 'removed from analysis')
						except ValueError:
							print('Audio file removed previously')

	return labels 


def generate_label_doc(config, labels_filename = 'labels.csv'):
	#creating header
	header = ['Filepath_label' + '\t' + 'Filepath_audio' + '\t' + 'Filename_audio' + '\t' + 'Filetype_audio' + '\t' + 'Selection' + '\t' + 'Start_time' + '\t' + 'End_time' + '\t' + 'Low_freq' + '\t' + 'High_freq' + '\t' + 'Species' + '\t' + 'File_id']


	#getting files/filepaths in labels and audio folders
	filepaths_label, files_label = get_files_and_paths(config['filepath_inputs']['label'], file_type = config['labels']['file_type'])
	filepaths_audio, files_audio = get_files_and_paths(config['filepath_inputs']['audio'], file_type = config['audio']['file_type'])
	# print('filepaths:')
	# print(filepaths_label)
	# print(filepaths_audio[0])
	#getting audio and label info
	labels = header
	labels += get_info_labels(filepaths_label, files_label, filepaths_audio, files_audio, config)

	create_folder(config['filepath_outputs']['label'])
	#tranforming to dataframe for easy saving as csv
	df = pd.DataFrame(labels)
	# print(df)
	df.to_csv(os.path.join(config['filepath_outputs']['label'], 'labels.csv'), header = None, index=False)
	df = pd.read_csv(os.path.join(config['filepath_outputs']['label'], 'labels.csv'), sep = '\t')
	# print(df)
	new_header = header[0].split('\t')
	new_header.extend(['Split_id', 'Final_filename', 'Split_start_time', 'Split_end_time', 'Vocal_start_time', 'Vocal_end_time'])
	new_df = pd.DataFrame(columns = new_header)
	for index, row in df.iterrows():
		# print('----------------------------')
		# print(row)
		# print(type(row))
		values = row.to_list()
		# print('val:', values)
		audio_duration = config['audio']['file_duration']
		splits = label_audio_splitter(config['audio_split']['split_type'], config['audio_split']['split_interval'], row['Start_time'], row['End_time'], audio_duration)
		for i, split in enumerate(splits):
			# print('val2: ', values)
			aux = values.copy()
			aux.append(str(i))
			aux.append(row['Filename_audio'] + '_selec_' + str(row['Selection']) + '_split_' + str(i))
			aux.extend(split)
			# print('values: ', aux)
			new_df.loc[len(new_df)] = aux
			# print(new_df)

		# print('----------------------------')
	for index, row in new_df.iterrows():
		new_df.at[index, 'File_id'] = index
	#tranforming to dataframe for easy saving as csv
	# df = pd.DataFrame(labels)

	
	new_df.to_csv(os.path.join(config['filepath_outputs']['label'], labels_filename), index=False)

# config_file = 'config.yaml'

# 	#Defining parameters

# with open(config_file) as f:
# 	cfg = yaml.load(f, Loader=yaml.FullLoader)
# generate_label_doc(cfg, 'labels_intersection.csv')
# config = settings.Settings_config(cfg)
# generate_label_doc(config)