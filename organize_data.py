import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
import pandas as pd
import shutil
import pdb
import sys
import yaml
from utils import *
import random
from dataset_cartography import *
import json

import numpy as np
import random
from sklearn.model_selection import StratifiedKFold

# config_file = sys.argv[1]
# #Defining parameters
# with open(config_file) as f:
#	 cfg = yaml.load(f, Loader=yaml.FullLoader)



# techniques = cfg['techniques']['selected']
# filepath_input = cfg['filepath_inputs']['audio_organized_vit']
# filepath_output = cfg['filepath_outputs']['audio_organized_vit']
# filepath_labels = cfg['filepath_outputs']['label']

'''
def stratified_split(X, y, class_percentages, random_seed=42):
	"""
	Split the dataset into train and test sets based on class percentages, while keeping the class distribution.
	
	Parameters:
	- X (np.ndarray): Features of the dataset (n_samples, n_features).
	- y (np.ndarray): Labels of the dataset (n_samples).
	- class_percentages (list): List of percentages for each class.
	- random_seed (int, optional): Random seed for reproducibility.
	
	Returns:
	- X_train (np.ndarray): Training features.
	- X_test (np.ndarray): Testing features.
	- y_train (np.ndarray): Training labels.
	- y_test (np.ndarray): Testing labels.
	"""
	
	# Set the random seed for reproducibility
	if random_seed is not None:
		random.seed(random_seed)
		np.random.seed(random_seed)
	
	# Get the unique classes
	unique_classes = np.unique(y)
	
	# Arrays to hold the train and test data
	X_train, X_test, y_train, y_test = [], [], [], []
	
	# Split data for each class
	for cls, percentage in zip(unique_classes, class_percentages):
		# Get indices of samples of the current class
		class_indices = np.where(y == cls)[0]
		
		# Shuffle the indices
		np.random.shuffle(class_indices)
		
		# Calculate the number of samples for training and testing
		num_samples = len(class_indices)
		num_train = int(num_samples * (percentage / 100))
		num_test = num_samples - num_train
		
		# Split the indices
		train_indices = class_indices[:num_train]
		test_indices = class_indices[num_train:]
		
		# Append the samples to the train and test sets
		X_train.append(X[train_indices])
		X_test.append(X[test_indices])
		y_train.append(y[train_indices])
		y_test.append(y[test_indices])
	
	# Convert lists to numpy arrays
	X_train = np.vstack(X_train)
	X_test = np.vstack(X_test)
	y_train = np.concatenate(y_train)
	y_test = np.concatenate(y_test)
	
	return X_train, X_test, y_train, y_test
'''

'''
def stratified_split(X, y, class_quantities, random_seed=42):
	"""
	Split the dataset into train and test sets based on direct class quantities, while keeping the class distribution.
	
	Parameters:
	- X (np.ndarray): Features of the dataset (n_samples, n_features).
	- y (np.ndarray): Labels of the dataset (n_samples).
	- class_quantities (list): List of quantities (number of samples) for each class for the training set.
	- random_seed (int, optional): Random seed for reproducibility.
	
	Returns:
	- X_train (np.ndarray): Training features.
	- X_test (np.ndarray): Testing features.
	- y_train (np.ndarray): Training labels.
	- y_test (np.ndarray): Testing labels.
	"""
	
	# Set the random seed for reproducibility
	if random_seed is not None:
		random.seed(random_seed)
		np.random.seed(random_seed)
	
	# Get the unique classes
	unique_classes = np.unique(y)
	
	# Arrays to hold the train and test data
	X_train, X_test, y_train, y_test = [], [], [], []
	
	# Split data for each class
	for cls, quantity in zip(unique_classes, class_quantities):
		# Get indices of samples of the current class
		class_indices = np.where(y == cls)[0]
		
		# Shuffle the indices
		np.random.shuffle(class_indices)
		
		# Calculate the number of samples for training and testing
		num_samples = len(class_indices)
		num_train = min(quantity, num_samples)  # Ensure that we don't take more than available
		num_test = num_samples - num_train
		
		# Split the indices
		train_indices = class_indices[:num_train]
		test_indices = class_indices[num_train:]
		
		# Append the samples to the train and test sets
		X_train.append(X[train_indices])
		X_test.append(X[test_indices])
		y_train.append(y[train_indices])
		y_test.append(y[test_indices])
	
	# Convert lists to numpy arrays
	X_train = np.vstack(X_train)
	X_test = np.vstack(X_test)
	y_train = np.concatenate(y_train)
	y_test = np.concatenate(y_test)
	
	return X_train, X_test, y_train, y_test
'''
'''
def stratified_split(X, y, class_quantities, random_seed=42):
	"""
	Split the dataset into train and test sets based on direct class quantities, while keeping the class distribution.
	
	Parameters:
	- X (np.ndarray): Features of the dataset (n_samples, n_features).
	- y (np.ndarray): Labels of the dataset (n_samples).
	- class_quantities (dict): Dictionary of class quantities, with class labels as keys and sample quantities as values for the training set.
	- random_seed (int, optional): Random seed for reproducibility.
	
	Returns:
	- X_train (np.ndarray): Training features.
	- X_test (np.ndarray): Testing features.
	- y_train (np.ndarray): Training labels.
	- y_test (np.ndarray): Testing labels.
	"""
	
	# Set the random seed for reproducibility
	if random_seed is not None:
		random.seed(random_seed)
		np.random.seed(random_seed)
	
	# Get the unique classes
	unique_classes = np.unique(y)
	
	# Arrays to hold the train and test data
	X_train, X_test, y_train, y_test = [], [], [], []
	
	# Split data for each class
	for cls in unique_classes:
		# Get indices of samples of the current class
		class_indices = np.where(y == cls)[0]
		
		# Shuffle the indices
		np.random.shuffle(class_indices)
		
		# Get the quantity for this class from the dictionary
		quantity = class_quantities.get(cls, 0)
		
		# Calculate the number of samples for training and testing
		num_samples = len(class_indices)
		num_train = min(quantity, num_samples)  # Ensure that we don't take more than available
		num_test = num_samples - num_train
		
		# Split the indices
		train_indices = class_indices[:num_train]
		test_indices = class_indices[num_train:]
		
		# Append the samples to the train and test sets
		X_train.append(X[train_indices])
		X_test.append(X[test_indices])
		y_train.append(y[train_indices])
		y_test.append(y[test_indices])
	
	# Convert lists to numpy arrays
	X_train = np.vstack(X_train)
	X_test = np.vstack(X_test)
	y_train = np.concatenate(y_train)
	y_test = np.concatenate(y_test)
	
	return X_train, X_test, y_train, y_test
'''

import numpy as np
import random

import numpy as np
import random

def stratified_split(X, y, train_class_quantities, test_class_quantities, random_seed=42):
	"""
	Split the dataset into train, test, and validation sets based on direct class quantities, while keeping the class distribution.
	
	Parameters:
	- X (np.ndarray): Features of the dataset (n_samples, n_features).
	- y (np.ndarray): Labels of the dataset (n_samples).
	- train_class_quantities (dict): Dictionary of class quantities for the training set (class labels as keys and sample quantities as values).
	- test_class_quantities (dict): Dictionary of class quantities for the testing set (class labels as keys and sample quantities as values).
	- random_seed (int, optional): Random seed for reproducibility.
	
	Returns:
	- X_train (np.ndarray): Training features.
	- X_test (np.ndarray): Testing features.
	- X_val (np.ndarray): Validation features.
	- y_train (np.ndarray): Training labels.
	- y_test (np.ndarray): Testing labels.
	- y_val (np.ndarray): Validation labels.
	"""
	print('##########################')
	print(X)
	print('##########################')
	print(y)
	print('##########################')
	# Set the random seed for reproducibility
	if random_seed is not None:
		random.seed(random_seed)
		np.random.seed(random_seed)
	
	# Get the unique classes
	unique_classes = np.unique(y)
	# print('unique_classes: ', unique_classes)
	# unique_classes.sort()
	# print('unique_classes sorted: ', unique_classes)

	
	# Arrays to hold the train, test, and validation data
	X_train, X_test, X_val, y_train, y_test, y_val = [], [], [], [], [], []
	
	# Split data for each class
	for cls in unique_classes:
		# Get indices of samples of the current class
		class_indices = np.where(y == cls)[0]
		print(class_indices)
		# Shuffle the indices
		np.random.shuffle(class_indices)
		
		# Get the quantities for this class from the dictionaries
		train_quantity = train_class_quantities.get(cls, 0)
		test_quantity = test_class_quantities.get(cls, 0)
		
		# Total number of samples in this class
		num_samples = len(class_indices)
		print(train_quantity, test_quantity, num_samples)
		# Calculate the number of samples for training and testing
		num_train = min(train_quantity, num_samples)  # Ensure that we don't take more than available
		num_test = min(test_quantity, num_samples - num_train)  # Ensure we don't take more than available
		num_val = num_samples - num_train - num_test  # The rest are allocated to validation
		print(num_train, num_test, num_val)



		# Split the indices
		train_indices = class_indices[:num_train]
		test_indices = class_indices[num_train:num_train + num_test]
		val_indices = class_indices[num_train + num_test:]
		print(train_indices, test_indices, val_indices)
		print(f"Type of X: {type(X)}")

		# counter = 0
		# for index, row in X.iterrows():
		# 	counter += 1
		# 	print('@@@@@@@@@@@@@@@@@@')
		# 	print(row)
		# 	print('-------------------')
		# 	print(index)
		# 	print('-------------------')
		# 	print(X.iloc[index])
		# 	if(counter > 175):
		# 		break
		# Append the samples to the train, test, and validation sets
		X_train.append(X.iloc[train_indices])
		X_test.append(X.iloc[test_indices])
		X_val.append(X.iloc[val_indices])
		
		y_train.append([cls] * num_train)
		y_test.append([cls] * num_test)
		y_val.append([cls] * num_val)
	
	# Convert lists to numpy arrays
	X_train = np.vstack(X_train)
	X_test = np.vstack(X_test)
	X_val = np.vstack(X_val)
	
	y_train = np.concatenate(y_train)
	y_test = np.concatenate(y_test)
	y_val = np.concatenate(y_val)
	
	return X_train, X_test, X_val, y_train, y_test, y_val



def stratified_Kfold_split(k, x, y, val_percentage, folds_filepath):
	x0 = [item[0] for item in x]

	skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
	skf.get_n_splits(x0, y)

	df = pd.DataFrame({'Filename': x0, 'Species': y})

	random.seed(42)

	fold_splits = []
	for i, (train_index, test_index) in enumerate(skf.split(x0, y)):
		print(f"Fold {i}:")
		print(f"  Train: index={train_index}")
		print(f"  Test:  index={test_index}")
		# random.shuffle(train_index)
		# Print the shuffled list
		# print(train_index)
		# random.shuffle(train_index)

		# print(train_index)

		new_X = [x[i] for i in train_index]
		new_y = [y[i] for i in train_index]
		x_train, x_val, y_train, y_val = train_test_split(new_X, new_y, test_size=val_percentage, stratify = new_y, random_state = 42)
		x_test = [x[i] for i in test_index]
		y_test = [y[i] for i in test_index]

		print(x_train, x_val, y_train, y_val)
		df['fold' + str(i)] = df['Filename'].map(lambda x: 'train' if x in x_train else ('val' if x in x_val else ('test' if x in x_test else '')))

		fold_splits.append(['fold' + str(i), x_train, y_train, x_test, y_test, x_val, y_val])
	df.to_csv(folds_filepath, index=False)
	return fold_splits

def data_splits_dataset_cartography(filenames, labels, filenames_DC, labels_DC):
	selected_indices = np.isin(filenames, filenames_DC)
	remaining_indices = ~selected_indices



'''
# filepath_input_data = "/home/gustavo/Documentos/Dados/Gerados/" + technique + '_imgs'
# filepath_output_data = "/home/gustavo/Documentos/Dados/Gerados/teste_vit_input_" + technique
# labels = sys.argv[2]#"C:/Users/GustavoLopes/Documents/Dados/Gerados/new_labels_12species.csv"

# print("OUTPUT PATH: ", filepath_output_data)

# try:
# 	os.mkdir(filepath_output_data)
# except:
# 	pass

df = pd.read_csv(os.path.join(filepath_labels, 'labels.csv'), sep='\t')  
df = df[['Filename_audio', 'Species', 'Selection', 'File_id']]
# print(df)

labels = df['Species'].unique()

# print(labels)

data = []
for index, row in df.iterrows():
	# print(row['Filename_audio'])
	data.append([row['Filename_audio'].split('.')[0] + '_selec_' + str(row['Selection']), str(row['Species']), str(row['File_id'])])
new_df = pd.DataFrame(data)
# new_df.rename(columns={0: "file", 1: "label"})
headers = ['Filename_audio', 'Species', 'File_id']
new_df.columns = headers


for technique in techniques:
	_, filename_input = get_files_and_paths(os.path.join(filepath_input, technique, 'imgs'), file_type = 'png')
	# print('--------------------')
	# print(filepath_input)
	x = []
	y = []
	# breakpoint()
	for i, file in enumerate(filename_input):
		# print('--------------------')
		print(file)
		filename_df = file.split('_split')[0]
		print(filename_df)
		ind = new_df.index[new_df['Filename_audio'].str.startswith(filename_df)].to_list()
		print(ind)
		for indice in ind:
			x.append([file, new_df['File_id'][indice].item()])
			y.append(new_df.loc[indice]['Species'].item())

	print(x)
	print(y)
	my_dict = {i:y.count(i) for i in y}
	

	X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=1)

	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

	
val = 0
val = create_direc(X_train, y_train, 'train', val)
val = create_direc(X_test, y_test, 'test', val)
val = create_direc(X_val, y_val, 'val', val)
'''
def value_exists_in_dict(value, dictionary):
	# Iterate over each key-value pair in the dictionary
	for key, val_list in dictionary.items():
		# Check if the value exists in the list
		if value in val_list:
			return True
	return False

def get_quantity_per_key(dictionary):
	# Create a new dictionary to store the quantity (length of list) for each key
	quantity_dict = {key: len(val_list) for key, val_list in dictionary.items()}
	return quantity_dict

def subtract_dicts(dict1, dict2):
	# Subtract the values of dict2 from dict1, and store the result in a new dictionary
	result = {key: dict1.get(key, 0) - dict2.get(key, 0) for key in dict1}
	
	# Add any keys from dict2 that are not in dict1 (with negative values)
	for key in dict2:
		if key not in dict1:
			result[key] = -dict2[key]
	
	return result

def apply_percentage_to_dict(input_dict, percentage):
	"""
	Apply a percentage reduction to each value in the dictionary with lists or single integers as values.
	
	Parameters:
	- input_dict (dict): Dictionary where the values are either single integers or lists of integers.
	- percentage (float): Percentage to apply to each value (e.g., 80 for 80%).
	
	Returns:
	- new_dict (dict): Dictionary with the same keys, but the values reduced by the given percentage.
	"""
	# Step 1: Calculate the percentage of each value in the lists or individual values
	percentage_dict = {}
	for key, values in input_dict.items():
		# If values is a list, apply percentage to each item in the list
		if isinstance(values, list):
			percentage_dict[key] = [int(v * (percentage / 100)) for v in values]
		# If values is a single integer, apply percentage directly
		elif isinstance(values, int):
			percentage_dict[key] = int(values * (percentage / 100))
	
	return percentage_dict

def prepare_data_and_split(filepath_labels, filename_labels, filepath_input, techniques, DC_indices, random_seed=42):
	# Step 1: Read the label data and prepare the DataFrame
	df = pd.read_csv(os.path.join(filepath_labels, filename_labels), sep=',')
	# df = df[['Filename_audio', 'Species', 'Selection', 'File_id']]
	print(df)
	labels = df['Species'].unique()
	total_count_species = df['Species'].value_counts()
	total_count_species_dict = total_count_species.to_dict()
	print("total_count_species: ", total_count_species)
	print("total_count_species_dict: ", total_count_species_dict)

	DC_indices_quant = get_quantity_per_key(DC_indices)
	print("DC_indices_quant: ", DC_indices_quant)

	print("*******************************************************")

	total_count_x_train_in_val = apply_percentage_to_dict(total_count_species_dict, 60)
	print("total_count_x_train_in_val: ", total_count_x_train_in_val)
	total_count_x_test_in_val = apply_percentage_to_dict(total_count_species_dict, 20)
	print("total_count_x_test_in_val: ", total_count_x_test_in_val)

	print("*******************************************************")

	total_count_x_test_in_train = apply_percentage_to_dict(total_count_species_dict, 20)
	print("total_count_x_test_in_train: ", total_count_x_test_in_train)
	total_count_x_train_in_train = apply_percentage_to_dict(total_count_species_dict, 60)
	total_count_x_train_in_train = subtract_dicts(total_count_x_train_in_train, DC_indices_quant)
	print("total_count_x_train_in_train: ", total_count_x_train_in_train)

	print("*******************************************************")
	total_count_x_train_in_test = apply_percentage_to_dict(total_count_species_dict, 60)
	print("total_count_x_train_in_test: ", total_count_x_train_in_test)
	total_count_x_test_in_test = apply_percentage_to_dict(total_count_species_dict, 20)
	total_count_x_test_in_test = subtract_dicts(total_count_x_test_in_test, DC_indices_quant)
	print("total_count_x_test_in_test: ", total_count_x_test_in_test)

	print("*******************************************************")

	removed_DC_count_species_dict = subtract_dicts(total_count_species_dict, DC_indices_quant)
	removed_DC_count_x_train = apply_percentage_to_dict(removed_DC_count_species_dict, 60)
	print("removed_DC_count_x_train: ", removed_DC_count_x_train)
	removed_DC_count_x_test = apply_percentage_to_dict(removed_DC_count_species_dict, 20)
	print("removed_DC_count_x_test: ", removed_DC_count_x_test)

	# Prepare data for splitting
	# data = []
	# for index, row in df.iterrows():
	#	 data.append([row['Filename_audio'].split('.')[0] + '_selec_' + str(row['Selection']),
	#				  str(row['Species']), str(row['File_id'])])
	# new_df = pd.DataFrame(data, columns=['Filename_audio', 'Species', 'File_id'])
	DC_x, DC_y = [], []
	# Step 2: Loop through the techniques and prepare data for each technique
	for technique in techniques:
		_, filename_input = get_files_and_paths(os.path.join(filepath_input, technique, 'imgs'), file_type='png')

		x = []
		y = []
		
		for i, file in enumerate(filename_input):
			filename_df = file.split('.png')[0]
			# print(filename_df)
			ind = df.index[df['Final_filename'].str.startswith(filename_df)].to_list()
			# print(ind)

			for indice in ind:
				# print(indice)
				if(value_exists_in_dict(indice, DC_indices)):
					DC_x.append([file, df['File_id'][indice]])
					DC_y.append(df.loc[indice]['Species'])
				else:
					x.append([file, df['File_id'][indice]])
					y.append(df.loc[indice]['Species'])

		x = np.array(x)
		y = np.array(y)
		print(x)
		print('##################')
		print(y)
		# # Step 3: Call stratified_split to get the train, test, and validation sets
		X_train, X_test, X_val, y_train, y_test, y_val = stratified_split(x, y, total_count_x_train_in_val, total_count_x_test_in_val, random_seed)
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(X_train)
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(X_test)
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(X_val)
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(y_train)
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(y_test)
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print(y_val)

		# # Print the results for each split
		# print(f"Technique: {technique}")
		# print(f"Train Set Size: {len(X_train)}")
		# print(f"Test Set Size: {len(X_test)}")
		# print(f"Validation Set Size: {len(X_val)}")
		# print(f"Class distribution in train set: {dict(zip(*np.unique(y_train, return_counts=True)))}")
		# print(f"Class distribution in test set: {dict(zip(*np.unique(y_test, return_counts=True)))}")
		# print(f"Class distribution in validation set: {dict(zip(*np.unique(y_val, return_counts=True)))}")

def convert_dict_values(d):
	for key, value in d.items():
		if isinstance(value, list):
			d[key] = [int(v) if isinstance(v, np.int64) else v for v in value]
	return d

'''
caminho_csv = '../data/generated_data/vis/vocal_vmd.csv'  # Substitua pelo caminho do seu arquivo CSV
threshold = 0.9  # Defina o threshold desejado
df = pd.read_csv(caminho_csv)
# print(df)

df['filename'] = df['filename'].str.replace(r'\.wav$', '', regex=True)
labels = pd.read_csv('../data/generated_data/labels_LEEC.csv').loc[:, ['Species', 'Final_filename']]
print(labels[30:40])

df = pd.merge(df, labels, left_on=['filename'], right_on=['Final_filename'], how='left').drop('Final_filename', axis=1)
print(df[30:40])
df['is_PM'] = extract_time_of_day(df)

loc = extract_location(df)
# print(loc)
df['aa'] = 0
df['br'] = 0
df['ma'] = 0

# print(df[['original_filename', 'aa', 'br', 'ma', 'is_PM']])
for i, location in enumerate(loc):
	# print(i, location)
	df.loc[i, location] = 1
# print(df[['filename', 'aa', 'br', 'ma', 'is_PM']])

_, colunas_removidas = remove_correlated_columns(df.drop('Species', axis=1), threshold)


# # Exibir as colunas removidas e o novo DataFrame
# print("Colunas removidas:", colunas_removidas)
# print("\nNovo DataFrame após remoção das colunas correlacionadas:")
# print(df)
df = df.drop(colunas_removidas, axis=1)
# print(df)
# df.to_csv('indices_normalizados.csv', index=False)
df.to_csv('indices_normalizados_loc_time.csv', index=False)

#######################################

classes = ['basi_culi', 'myio_leuc', 'vire_chiv', 'cycl_guja', 'pita_sulp', 'zono_cape', 'dend_minu', 'apla_leuc', 'isch_guen', 'phys_cuvi', 'boan_albo', 'aden_marm']
classes = classes#[0:7]
dic_classes = {}

for i, item in enumerate(classes):
	dic_classes[item] = i
# print(dic_classes)

print(df)
print('########################')
print(df['Species'].isin(classes)[30:40])
print('########################')
df = df[df['Species'].isin(classes)]
wrong_species_df = df[~df['Species'].isin(classes)]
print('wrong_species_df: \n', wrong_species_df)
print(df)
print(df['Species'].value_counts())

y_train = df.loc[:,"Species"]
x_train = df.drop(columns = ['filename', 'Species'])
print('x_train: ', x_train)
y_train = y_train.map(dic_classes)
# y_test = y_test.map(dic_classes)
print(y_train)
print(y_train.value_counts())

result = run_experiment(x_train, y_train, EPOCHS = 20)

DC_indices = generate_DC_indices(df, result, x_train, METRIC_NAME = 'f1')

DC_indices = convert_dict_values(DC_indices)

with open('DC_indices_vmd.txt', 'w') as f:
	json.dump(DC_indices, f)


with open('DC_indices_vmd.txt', 'r') as f:
	DC_indices = json.load(f)

prepare_data_and_split('../data/generated_data/', 'labels_LEEC.csv', '../data/generated_data/transformed', ['vmd'], DC_indices)
'''
'''
df = pd.read_csv('../data/generated_data/labels_LEEC.csv')
df = df[df['Filepath_audio'] != 'File not found!']
num_labels = df['Species'].unique()
total_count_species = df['Species'].value_counts()
print(total_count_species)
sorted_species_counts = total_count_species.sort_index()
print(sorted_species_counts)
counts_list = sorted_species_counts.to_dict()
print(counts_list)
train_size = apply_percentage_to_dict(counts_list, 60)
test_size = apply_percentage_to_dict(counts_list, 20)
print(train_size, test_size)


x = df[['Final_filename', 'File_id']]
y = df['Species']

X_train, X_test, X_val, y_train, y_test, y_val = stratified_split(x, y, train_size, test_size)
print(X_test)
print(y_test)
'''

def create_direc(x, y, dir_type, in_path, out_path):
	
	create_folder(os.path.join(out_path, dir_type))
	for label in np.unique(y):
		create_folder(os.path.join(out_path, dir_type, label))

	df = pd.DataFrame(x)
	df.rename(columns={0:'file', 1:'id'}, inplace=True)
	df['label'] = y
	print("DataFrame:")
	print(df)
	# print(y)
	# df.to_csv(filepath_output_data + '/' + dir_type + '/' + dir_type + '.csv', index=False)
	# i = val
	for _, row in df.iterrows():
		# print("OUTPUT PATH: ", filepath_output)
		# print(str(row['file']))
		# print(row)
		dest_fpath = os.path.join(out_path, dir_type, str(row['label']), str(row['id']) + '.png')
		origin_fpath = os.path.join(in_path, str(row['file']) + '.png')
		print("DESTINATION PATH: ", dest_fpath)
		print("ORIGIN PATH: ", origin_fpath)
		# os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
		shutil.copyfile(origin_fpath, dest_fpath)
		# os.rename(dest_fpath, os.path.join(str(filepath_output), technique, str(dir_type), str(row['label']), str(row['id']) + '.png'))
		# i += 1
	# return i
'''
techniques = ['none', 'vmd', 'towsey', 'spectral_subtraction', 'pcen']

for technique in techniques:
	create_direc(X_train, y_train, 'train', '/data/generated_data/transformed/' + technique + '/imgs_cropped_vocal/', '/data/generated_data/network_inputs/' + technique)
	create_direc(X_test, y_test, 'test', '/data/generated_data/transformed/' + technique + '/imgs_cropped_vocal/', '/data/generated_data/network_inputs/' + technique)
	create_direc(X_val, y_val, 'val', '/data/generated_data/transformed/' + technique + '/imgs_cropped_vocal/', '/data/generated_data/network_inputs/' + technique)

'''
'''
#### ORIGINAL CODE FOR TECHNIQUES #####
# df = pd.read_csv('labels_LEEC.csv')
df = pd.read_csv('../data/generated_data/labels_LEEC.csv')
df = df[df['Filepath_audio'] != 'File not found!']
x = df[['Final_filename', 'File_id']].values.tolist()
y = df['Species'].tolist()
# fold_splits = stratified_Kfold_split(5, x, y, 1/8, 'folds.csv')
fold_splits = stratified_Kfold_split(5, x, y, 1/8, '../data/generated_data/folds.csv')

techniques = ['none', 'vmd', 'towsey', 'spectral_subtraction', 'pcen', 'nmf']
audio_cut_type = 'imgs'

for fold in fold_splits:
	for technique in techniques:
		create_direc(fold[1], fold[2], 'train', '/data/generated_data/transformed/' + technique + '/' + audio_cut_type + '/', '/data/generated_data/network_inputs/' + technique + '/' + audio_cut_type + '/' + fold[0])
		create_direc(fold[3], fold[4], 'test', '/data/generated_data/transformed/' + technique + '/' + audio_cut_type + '/', '/data/generated_data/network_inputs/' + technique + '/' + audio_cut_type + '/' + fold[0])
		create_direc(fold[5], fold[6], 'val', '/data/generated_data/transformed/' + technique + '/' + audio_cut_type + '/', '/data/generated_data/network_inputs/' + technique + '/' + audio_cut_type + '/' + fold[0])
'''

# df = pd.read_csv('labels_LEEC.csv')
print('opa')
df = pd.read_csv('../data/generated_data/labels_LEEC.csv')
df_excluded = df[df['Filepath_audio'] == 'File not found!']
df = df[df['Filepath_audio'] != 'File not found!']
# excluded_list_index = df_excluded.index.tolist()

# _, noisy_filenames = generate_DC_indices_vit(path, 'log_info_gustavo_pred_', 'log_info_gustavo_ids_', 'log_info_gustavo_labels_', 'log_info_gustavo_probs_')
# _, noisy_filenames = get_label_noise_indexes('.\\vis\\indices', 'vocal_vmd.csv', df)
# DC_vit:
'''
noisy_filenames = ['LEEC41__0__20161113_205400_ma_selec_1_split_0', 'LEEC36__0__20161228_192400_br_selec_4_split_0', 'LEEC21__0__20170110_204600_br_selec_1_split_0', 'LEEC36__0__20161214_221600_aa_selec_1_split_1', 'LEEC36__0__20161214_221600_aa_selec_4_split_1', 'LEEC49__0__20170115_221600_br_selec_1_split_0', 'LEEC42__0__20161123_213900_aa_selec_1_split_0', 'LEEC49__0__20161120_200900_aa_selec_1_split_0', 'LEEC45__1__20170119_221600_br_selec_1_split_0', 'LEEC45__1__20170119_221600_br_selec_5_split_1', 'LEEC48__0__20161126_191600_aa_selec_1_split_0', 'LEEC45__0__20161019_200900_ma_selec_1_split_1', 'LEEC49__0__20170117_200100_br_selec_1_split_0', 'LEEC03__0__20161129_204600_aa_selec_2_split_0', 'LEEC33__0__20161231_213100_br_selec_2_split_0', 'LEEC33__0__20161231_213100_br_selec_5_split_0', 'LEEC48__0__20170126_200900_br_selec_1_split_0', 'LEEC49__0__20161023_200900_ma_selec_1_split_0', 'LEEC40__0__20161230_222400_br_selec_3_split_0', 'LEEC43__0__20161022_205400_ma_selec_2_split_1', 'LEEC06__0__20161206_221600_aa_selec_2_split_1', 'LEEC36__0__20170115_200100_br_selec_2_split_0', 'LEEC43__0__20161103_213900_ma_selec_1_split_1', 'LEEC18__1__20161230_213900_br_selec_7_split_0', 'LEEC42__0__20161113_221600_ma_selec_2_split_0', 'LEEC48__0__20161026_204600_ma_selec_2_split_1', 'LEEC36__0__20161217_192400_aa_selec_2_split_0', 'LEEC42__0__20170105_183100_br_selec_1_split_1', 'LEEC08__0__20170111_222400_br_selec_2_split_0', 'LEEC33__0__20170104_204600_br_selec_2_split_0', 'LEEC05__0__20161207_221600_aa_selec_1_split_0', 'LEEC36__0__20161122_204600_aa_selec_8_split_0', 'LEEC43__0__20161106_200900_ma_selec_1_split_0', 'LEEC43__0__20161106_200900_ma_selec_1_split_1', 'LEEC05__0__20161103_183900_ma_selec_3_split_0', 'LEEC06__0__20161116_200900_ma_selec_1_split_0', 'LEEC40__0__20170109_213100_br_selec_4_split_0', 'LEEC48__0__20161125_200900_aa_selec_1_split_0', 'LEEC06__0__20170123_205400_br_selec_3_split_0', 'LEEC45__0__20161108_213900_ma_selec_1_split_0', 'LEEC45__0__20161108_213900_ma_selec_2_split_1', 'LEEC48__0__20161129_213100_aa_selec_1_split_1', 'LEEC08__0__20161117_205400_ma_selec_2_split_0', 'LEEC06__0__20161203_213900_aa_selec_1_split_0', 'LEEC26__0__20170113_192400_br_selec_2_split_0', 'LEEC08__0__20161203_200900_aa_selec_1_split_2', 'LEEC36__0__20161206_192400_aa_selec_4_split_0', 'LEEC49__0__20161203_222400_aa_selec_14_split_0', 'LEEC42__0__20161116_204600_ma_selec_1_split_0', 'LEEC26__0__20161104_183100_ma_selec_1_split_1', 'LEEC48__0__20161229_222400_br_selec_5_split_0', 'LEEC06__0__20161207_221600_aa_selec_3_split_1', 'LEEC43__0__20161109_191600_ma_selec_1_split_0', 'LEEC49__0__20170118_183100_br_selec_1_split_0', 'LEEC49__0__20161127_191600_aa_selec_1_split_0', 'LEEC15__0__20170122_192400_br_selec_1_split_1', 'LEEC45__1__20170120_205400_br_selec_4_split_1', 'LEEC43__0__20170120_221600_br_selec_2_split_0', 'LEEC48__0__20161116_213100_ma_selec_1_split_0', 'LEEC06__0__20161102_192400_ma_selec_1_split_0', 'LEEC18__1__20161231_213100_br_selec_2_split_0', 'LEEC45__0__20161023_191600_ma_selec_2_split_0', 'LEEC48__0__20161207_213100_aa_selec_1_split_1', 'LEEC08__0__20161113_183900_ma_selec_1_split_0', 'LEEC48__0__20161102_221600_ma_selec_1_split_0', 'LEEC45__0__20161219_200100_aa_selec_1_split_2', 'LEEC36__0__20161113_213900_ma_selec_2_split_0', 'LEEC33__0__20170125_222400_br_selec_2_split_0', 'LEEC26__0__20170113_213900_br_selec_2_split_1', 'LEEC06__0__20170114_191600_br_selec_2_split_0', 'LEEC08__0__20170111_205400_br_selec_1_split_0', 'LEEC08__0__20161217_204600_aa_selec_1_split_0', 'LEEC43__0__20161024_183100_ma_selec_3_split_0', 'LEEC15__0__20170119_200100_br_selec_1_split_1', 'LEEC08__0__20161121_200900_aa_selec_1_split_0', 'LEEC48__0__20170109_213900_br_selec_2_split_0', 'LEEC23__1__20170125_204600_br_selec_1_split_0', 'LEEC45__1__20170102_200900_br_selec_4_split_0', 'LEEC48__0__20161103_183100_ma_selec_2_split_0', 'LEEC18__1__20161214_222400_aa_selec_1_split_0', 'LEEC05__0__20161106_192400_ma_selec_1_split_0', 'LEEC36__0__20161215_192400_aa_selec_2_split_1', 'LEEC06__0__20170124_213900_br_selec_1_split_1', 'LEEC03__0__20161231_213100_br_selec_5_split_1', 'LEEC08__0__20161231_192400_br_selec_1_split_0', 'LEEC08__0__20161026_221600_ma_selec_7_split_0', 'LEEC40__0__20170122_200100_br_selec_2_split_0', 'LEEC49__0__20161026_204600_ma_selec_10_split_0', 'LEEC45__1__20170104_200100_br_selec_6_split_0', 'LEEC08__0__20161104_213900_ma_selec_3_split_0', 'LEEC08__0__20161115_204600_ma_selec_2_split_1', 'LEEC40__0__20170118_205400_br_selec_4_split_0', 'LEEC33__0__20161230_221600_br_selec_1_split_0', 'LEEC33__0__20161230_221600_br_selec_2_split_1', 'LEEC23__1__20161213_184100_aa_selec_1_split_0', 'LEEC23__1__20161213_184100_aa_selec_2_split_0', 'LEEC48__0__20161217_200100_aa_selec_1_split_0', 'LEEC36__0__20161103_200100_ma_selec_1_split_0', 'LEEC06__0__20161230_183900_br_selec_1_split_0', 'LEEC45__0__20161129_221600_aa_selec_1_split_1', 'LEEC05__0__20170103_222400_br_selec_1_split_0', 'LEEC36__0__20161210_191600_aa_selec_3_split_0', 'LEEC09__0__20170126_204600_br_selec_4_split_1', 'LEEC02__0__20170110_204600_br_selec_2_split_0', 'LEEC05__0__20170118_205400_br_selec_1_split_0', 'LEEC33__0__20170103_191600_br_selec_2_split_0', 'LEEC36__0__20161207_184100_aa_selec_1_split_1', 'LEEC05__0__20170125_192400_br_selec_1_split_0', 'LEEC36__0__20170104_213100_br_selec_1_split_0', 'LEEC09__0__20161217_213100_aa_selec_2_split_0', 'LEEC49__0__20170118_213900_br_selec_2_split_0', 'LEEC33__0__20170109_222400_br_selec_1_split_1', 'LEEC18__1__20161204_184100_aa_selec_1_split_0', 'LEEC08__0__20170113_204600_br_selec_2_split_0', 'LEEC08__0__20161121_191600_aa_selec_2_split_0', 'LEEC49__0__20161213_204600_aa_selec_3_split_0', 'LEEC08__0__20161231_204600_br_selec_1_split_0', 'LEEC06__0__20161122_204600_aa_selec_3_split_1', 'LEEC45__0__20161123_213900_aa_selec_1_split_0', 'LEEC42__0__20170102_183900_br_selec_2_split_0', 'LEEC49__0__20161112_183900_ma_selec_3_split_0', 'LEEC08__0__20161217_205400_aa_selec_2_split_0', 'LEEC43__0__20170123_200900_br_selec_4_split_0', 'LEEC43__0__20161215_191600_aa_selec_2_split_0', 'LEEC45__1__20170121_192400_br_selec_2_split_0', 'LEEC02__0__20161208_183900_ma_selec_1_split_3', 'LEEC05__0__20161211_192400_aa_selec_1_split_0', 'LEEC18__0__20161111_051400_ma_selec_1_split_0', 'LEEC40__0__20161219_080100_aa_selec_7_split_0', 'LEEC33__0__20161222_055900_aa_selec_1_split_0', 'LEEC49__0__20161202_064400_aa_selec_1_split_0', 'LEEC03__0__20161119_055900_ma_selec_2_split_0', 'LEEC24__0__20161127_072900_aa_selec_9_split_0', 'LEEC03__0__20170111_080100_br_selec_1_split_0', 'LEEC48__0__20170102_055900_br_selec_3_split_0', 'LEEC24__0__20161031_071600_ma_selec_1_split_0', 'LEEC33__0__20161120_051400_aa_selec_2_split_0', 'LEEC48__0__20161221_050100_aa_selec_4_split_0', 'LEEC06__0__20161128_054600_aa_selec_3_split_0', 'LEEC09__0__20161123_081400_aa_selec_2_split_0', 'LEEC02__0__20161203_063100_ma_selec_4_split_0', 'LEEC36__0__20170126_072900_br_selec_7_split_0', 'LEEC05__0__20161202_071600_aa_selec_1_split_0', 'LEEC33__0__20161208_054600_aa_selec_3_split_0', 'LEEC05__0__20161120_081400_ma_selec_3_split_0', 'LEEC21__0__20170114_064400_br_selec_1_split_1', 'LEEC36__0__20170124_071600_br_selec_3_split_0', 'LEEC49__0__20170118_054600_br_selec_1_split_0', 'LEEC18__0__20170118_080100_br_selec_2_split_0', 'LEEC06__0__20170123_072900_br_selec_2_split_0', 'LEEC15__0__20161123_055900_aa_selec_1_split_0', 'LEEC09__0__20161206_055900_aa_selec_4_split_0', 'LEEC08__0__20161108_050100_ma_selec_3_split_0', 'LEEC26__0__20161023_055900_ma_selec_3_split_2', 'LEEC36__0__20161023_063100_ma_selec_1_split_0', 'LEEC36__0__20161023_063100_ma_selec_2_split_1', 'LEEC21__0__20161028_051400_ma_selec_7_split_0', 'LEEC33__0__20170126_063100_br_selec_4_split_0', 'LEEC09__0__20161218_054600_aa_selec_7_split_0', 'LEEC36__0__20161117_064400_ma_selec_1_split_0', 'LEEC36__0__20161116_054600_ma_selec_5_split_0', 'LEEC41__0__20161211_071600_aa_selec_3_split_0', 'LEEC40__0__20161230_071600_br_selec_1_split_0', 'LEEC23__0__20170124_054600_br_selec_2_split_0', 'LEEC21__0__20161020_054600_ma_selec_5_split_0', 'LEEC06__0__20161217_081400_aa_selec_4_split_0', 'LEEC02__0__20170116_054600_aa_selec_1_split_0', 'LEEC06__0__20161207_051400_aa_selec_8_split_0', 'LEEC40__0__20161221_072900_aa_selec_2_split_0', 'LEEC06__0__20161124_071600_aa_selec_4_split_0', 'LEEC15__0__20161026_080100_ma_selec_3_split_0', 'LEEC49__0__20161205_071600_aa_selec_7_split_0', 'LEEC26__0__20170108_063100_br_selec_5_split_0', 'LEEC48__0__20161020_055900_ma_selec_2_split_0', 'LEEC36__0__20161102_081400_ma_selec_5_split_0', 'LEEC49__0__20161123_054600_aa_selec_1_split_0', 'LEEC48__0__20161108_071600_ma_selec_6_split_0', 'LEEC49__0__20161029_064400_ma_selec_6_split_0', 'LEEC26__0__20161109_072900_ma_selec_1_split_0', 'LEEC48__0__20161104_051400_ma_selec_1_split_2', 'LEEC26__0__20161110_054600_ma_selec_2_split_0', 'LEEC36__0__20161107_071600_ma_selec_3_split_1', 'LEEC24__0__20161127_051400_aa_selec_5_split_0', 'LEEC24__0__20161127_051400_aa_selec_6_split_0', 'LEEC43__0__20161029_054600_ma_selec_3_split_0', 'LEEC09__0__20170102_054600_br_selec_2_split_1', 'LEEC49__0__20161125_055900_aa_selec_8_split_0', 'LEEC03__0__20170105_064400_br_selec_2_split_0', 'LEEC43__0__20161214_063100_aa_selec_3_split_0', 'LEEC09__0__20161204_064400_aa_selec_3_split_0', 'LEEC03__0__20161201_055900_aa_selec_2_split_1', 'LEEC06__0__20161121_055900_aa_selec_2_split_0', 'LEEC49__0__20161118_051400_ma_selec_4_split_1', 'LEEC49__0__20170121_072900_br_selec_2_split_0', 'LEEC08__0__20161026_081400_ma_selec_2_split_0', 'LEEC03__0__20161111_051400_ma_selec_4_split_0', 'LEEC36__0__20161026_071600_ma_selec_2_split_0', 'LEEC15__0__20161114_054600_ma_selec_4_split_0', 'LEEC41__0__20161114_064400_ma_selec_1_split_0', 'LEEC33__0__20161217_051400_aa_selec_3_split_0', 'LEEC36__0__20161111_081400_ma_selec_3_split_0', 'LEEC43__0__20161020_063100_ma_selec_3_split_0', 'LEEC08__0__20161023_064400_ma_selec_4_split_0', 'LEEC36__0__20161102_055900_ma_selec_5_split_2', 'LEEC36__0__20161102_055900_ma_selec_6_split_0', 'LEEC08__0__20161115_071600_ma_selec_1_split_0', 'LEEC03__0__20161123_054600_aa_selec_1_split_1', 'LEEC08__0__20170114_071600_br_selec_1_split_0', 'LEEC02__0__20170115_081400_aa_selec_6_split_0', 'LEEC33__0__20161110_050100_ma_selec_7_split_0', 'LEEC36__0__20161107_054600_ma_selec_1_split_0', 'LEEC26__0__20161121_080100_ma_selec_1_split_1', 'LEEC26__0__20161116_072900_ma_selec_2_split_0', 'LEEC03__0__20161112_055900_ma_selec_1_split_0', 'LEEC24__0__20161229_080100_br_selec_1_split_0', 'LEEC06__0__20161202_071600_aa_selec_1_split_0', 'LEEC24__0__20161019_063100_ma_selec_9_split_0', 'LEEC37__0__20161118_072900_ma_selec_2_split_0', 'LEEC40__0__20161107_064400_ma_selec_1_split_0', 'LEEC36__0__20161110_054600_ma_selec_1_split_0', 'LEEC43__0__20170123_071600_br_selec_3_split_0', 'LEEC09__0__20170113_063100_br_selec_1_split_0', 'LEEC49__0__20161216_054600_aa_selec_4_split_0', 'LEEC36__0__20161125_054600_aa_selec_1_split_0', 'LEEC03__0__20170109_063100_br_selec_3_split_0', 'LEEC26__0__20161121_054600_ma_selec_3_split_1', 'LEEC02__0__20170108_050100_aa_selec_4_split_0', 'LEEC09__0__20161214_051400_aa_selec_1_split_0', 'LEEC49__0__20161030_064400_ma_selec_2_split_0', 'LEEC18__0__20161128_064400_aa_selec_1_split_0', 'LEEC18__0__20161128_064400_aa_selec_8_split_0', 'LEEC21__0__20170107_071600_br_selec_4_split_0', 'LEEC21__0__20161209_064400_aa_selec_2_split_0', 'LEEC09__0__20161117_071600_ma_selec_7_split_0', 'LEEC40__0__20161030_064400_ma_selec_1_split_0', 'LEEC06__0__20161107_064400_ma_selec_9_split_0', 'LEEC43__0__20161116_072900_ma_selec_9_split_0', 'LEEC24__0__20161231_054600_br_selec_1_split_0', 'LEEC09__0__20161213_051400_aa_selec_1_split_0', 'LEEC49__0__20161209_055900_aa_selec_3_split_0', 'LEEC06__0__20170111_063100_br_selec_1_split_0', 'LEEC48__0__20161108_063100_ma_selec_1_split_0', 'LEEC49__0__20161115_071600_ma_selec_2_split_0', 'LEEC49__0__20170117_064400_br_selec_4_split_0', 'LEEC48__0__20161115_080100_ma_selec_5_split_0', 'LEEC06__0__20161208_050100_aa_selec_11_split_0', 'LEEC45__0__20170105_071600_br_selec_3_split_0', 'LEEC02__0__20161129_054600_ma_selec_11_split_0', 'LEEC24__0__20161203_063100_aa_selec_1_split_0', 'LEEC08__0__20161207_051400_aa_selec_2_split_0', 'LEEC40__0__20161129_081400_aa_selec_3_split_0', 'LEEC43__0__20161106_051400_ma_selec_3_split_0', 'LEEC06__0__20161113_071600_ma_selec_1_split_0', 'LEEC36__0__20170122_054600_br_selec_1_split_0', 'LEEC05__0__20161103_054600_ma_selec_3_split_1', 'LEEC09__0__20161222_051400_aa_selec_5_split_0', 'LEEC33__0__20170125_080100_br_selec_5_split_0', 'LEEC43__0__20161215_051400_aa_selec_5_split_0', 'LEEC49__0__20161112_080100_ma_selec_2_split_1', 'LEEC24__0__20161202_064400_aa_selec_4_split_0', 'LEEC09__0__20161219_055900_aa_selec_2_split_0', 'LEEC33__0__20170107_064400_br_selec_4_split_0', 'LEEC42__0__20161027_055900_ma_selec_5_split_0', 'LEEC06__0__20161024_071600_ma_selec_6_split_1', 'LEEC36__0__20161121_051400_aa_selec_7_split_0', 'LEEC18__0__20170118_051400_br_selec_5_split_0', 'LEEC23__0__20170122_055900_br_selec_1_split_0', 'LEEC03__0__20161127_054600_aa_selec_4_split_0', 'LEEC40__0__20161023_071600_ma_selec_8_split_0', 'LEEC26__0__20161110_055900_ma_selec_1_split_0', 'LEEC26__0__20161110_055900_ma_selec_3_split_0', 'LEEC09__0__20161110_072900_ma_selec_2_split_0', 'LEEC26__0__20161227_080100_aa_selec_1_split_0', 'LEEC40__0__20161201_071600_aa_selec_1_split_0', 'LEEC41__0__20161022_063100_ma_selec_3_split_0', 'LEEC21__0__20161202_081400_aa_selec_1_split_0', 'LEEC33__0__20161231_080100_br_selec_4_split_0', 'LEEC41__0__20170113_071600_br_selec_1_split_0', 'LEEC05__0__20161020_080100_ma_selec_1_split_0', 'LEEC06__0__20161116_050100_ma_selec_2_split_0', 'LEEC37__0__20161120_050100_aa_selec_1_split_0', 'LEEC06__0__20170108_064400_br_selec_3_split_0', 'LEEC06__0__20170108_064400_br_selec_5_split_0', 'LEEC42__0__20170102_064400_br_selec_5_split_0', 'LEEC26__0__20161022_072900_ma_selec_1_split_0', 'LEEC18__0__20161219_064400_aa_selec_1_split_0', 'LEEC21__0__20161107_071600_ma_selec_2_split_0', 'LEEC40__0__20161212_064400_aa_selec_1_split_0', 'LEEC06__0__20161023_055900_ma_selec_5_split_0', 'LEEC33__0__20161022_080100_ma_selec_2_split_0', 'LEEC24__0__20161201_055900_aa_selec_2_split_0', 'LEEC08__0__20161128_071600_aa_selec_5_split_0', 'LEEC43__0__20170115_072900_br_selec_4_split_0', 'LEEC36__0__20161122_071600_aa_selec_6_split_0', 'LEEC40__0__20161104_055900_ma_selec_4_split_0', 'LEEC09__0__20161231_063100_br_selec_1_split_0', 'LEEC06__0__20161213_051400_aa_selec_4_split_0', 'LEEC41__0__20161209_071600_aa_selec_3_split_0', 'LEEC09__0__20161024_055900_ma_selec_3_split_0', 'LEEC18__0__20161104_064400_ma_selec_1_split_0', 'LEEC45__0__20161212_081400_aa_selec_2_split_0', 'LEEC49__0__20161210_050100_aa_selec_1_split_0', 'LEEC09__0__20161210_080100_aa_selec_3_split_0', 'LEEC03__0__20161128_064400_aa_selec_5_split_0', 'LEEC49__0__20161117_064400_ma_selec_2_split_1', 'LEEC41__0__20161210_071600_aa_selec_1_split_0', 'LEEC41__0__20161210_071600_aa_selec_2_split_0', 'LEEC42__0__20161130_050100_aa_selec_1_split_0', 'LEEC43__0__20170106_063100_br_selec_2_split_0', 'LEEC06__0__20161219_055900_aa_selec_5_split_0', 'LEEC40__0__20161105_072900_ma_selec_4_split_0', 'LEEC40__0__20161105_072900_ma_selec_10_split_0', 'LEEC05__0__20161210_064400_aa_selec_3_split_0', 'LEEC18__0__20170114_064400_br_selec_2_split_0', 'LEEC21__0__20170101_081400_br_selec_3_split_0', 'LEEC49__0__20161207_063100_aa_selec_4_split_0', 'LEEC03__0__20161117_081400_ma_selec_3_split_0', 'LEEC09__0__20161213_080100_aa_selec_4_split_1', 'LEEC02__0__20161208_063100_ma_selec_3_split_0', 'LEEC49__0__20161120_071600_ma_selec_3_split_0', 'LEEC33__0__20161110_081400_ma_selec_3_split_0', 'LEEC49__0__20161117_071600_ma_selec_2_split_1', 'LEEC40__0__20161220_055900_aa_selec_4_split_0', 'LEEC05__0__20161026_081400_ma_selec_1_split_0', 'LEEC43__0__20161231_080100_br_selec_6_split_0', 'LEEC40__0__20161116_055900_ma_selec_5_split_0', 'LEEC40__0__20161116_055900_ma_selec_6_split_1', 'LEEC15__0__20161022_063100_ma_selec_1_split_0', 'LEEC03__0__20161103_054600_ma_selec_3_split_0', 'LEEC06__0__20161021_050100_ma_selec_2_split_0', 'LEEC02__0__20170108_071600_aa_selec_1_split_0', 'LEEC08__0__20161231_072900_br_selec_2_split_0', 'LEEC08__0__20161231_072900_br_selec_3_split_0', 'LEEC36__0__20161028_064400_ma_selec_4_split_0', 'LEEC33__0__20161219_081400_aa_selec_3_split_0', 'LEEC09__0__20161120_055900_aa_selec_3_split_0', 'LEEC09__0__20161204_055900_aa_selec_1_split_0', 'LEEC36__0__20161116_063100_ma_selec_15_split_0', 'LEEC03__0__20161118_063100_ma_selec_1_split_0', 'LEEC06__0__20161024_050100_ma_selec_4_split_1', 'LEEC24__0__20161120_050100_aa_selec_6_split_0', 'LEEC18__0__20161130_064400_aa_selec_1_split_0', 'LEEC41__0__20161214_063100_aa_selec_5_split_0', 'LEEC40__0__20170124_071600_br_selec_5_split_0', 'LEEC09__0__20161120_071600_aa_selec_2_split_0', 'LEEC48__0__20161027_081400_ma_selec_3_split_0', 'LEEC48__0__20161027_081400_ma_selec_4_split_0', 'LEEC42__0__20170117_064400_br_selec_2_split_0', 'LEEC33__0__20161117_064400_ma_selec_1_split_0', 'LEEC40__0__20161206_054600_aa_selec_2_split_0', 'LEEC42__0__20161024_071600_ma_selec_1_split_0', 'LEEC24__0__20161215_063100_aa_selec_2_split_0', 'LEEC33__0__20170126_081400_br_selec_1_split_0', 'LEEC33__0__20170126_081400_br_selec_1_split_1', 'LEEC41__0__20161215_072900_aa_selec_2_split_0', 'LEEC05__0__20161028_050100_ma_selec_3_split_1', 'LEEC03__0__20161118_080100_ma_selec_1_split_0', 'LEEC06__0__20161206_054600_aa_selec_4_split_0', 'LEEC15__0__20161030_055900_ma_selec_4_split_0', 'LEEC21__0__20161126_071600_aa_selec_3_split_0', 'LEEC08__0__20161105_055900_ma_selec_1_split_0', 'LEEC18__0__20161022_055900_ma_selec_1_split_2', 'LEEC08__0__20161020_051400_ma_selec_4_split_0', 'LEEC45__0__20161026_055900_ma_selec_1_split_0', 'LEEC42__0__20161123_054600_aa_selec_1_split_0', 'LEEC18__0__20161220_081400_aa_selec_2_split_0', 'LEEC02__0__20170110_055900_aa_selec_1_split_0', 'LEEC43__0__20161216_080100_aa_selec_2_split_0', 'LEEC06__0__20161024_064400_ma_selec_1_split_0', 'LEEC08__0__20161107_054600_ma_selec_1_split_0', 'LEEC37__0__20161019_051400_ma_selec_1_split_0', 'LEEC09__0__20161213_072900_aa_selec_6_split_0', 'LEEC41__0__20161215_055900_aa_selec_1_split_0', 'LEEC36__0__20170119_051400_br_selec_1_split_0', 'LEEC41__0__20161122_051400_aa_selec_3_split_0', 'LEEC41__0__20161205_055900_aa_selec_1_split_0', 'LEEC15__0__20161214_054600_aa_selec_1_split_0', 'LEEC43__0__20161021_063100_ma_selec_2_split_0', 'LEEC43__0__20161113_051400_ma_selec_3_split_0', 'LEEC18__0__20161202_054600_aa_selec_4_split_0', 'LEEC02__0__20170127_080100_aa_selec_3_split_0', 'LEEC49__0__20161104_081400_ma_selec_2_split_0', 'LEEC05__0__20170105_071600_br_selec_2_split_0', 'LEEC03__0__20161125_054600_aa_selec_4_split_0', 'LEEC26__0__20161118_063100_ma_selec_3_split_0', 'LEEC15__0__20161024_071600_ma_selec_1_split_0', 'LEEC33__0__20170108_064400_br_selec_3_split_0', 'LEEC48__0__20161112_072900_ma_selec_2_split_0', 'LEEC48__0__20161112_072900_ma_selec_3_split_0', 'LEEC45__0__20161111_064400_ma_selec_1_split_0', 'LEEC37__0__20161101_071600_ma_selec_2_split_0', 'LEEC06__0__20161120_051400_aa_selec_1_split_0', 'LEEC37__0__20161107_081400_ma_selec_10_split_0', 'LEEC23__0__20170126_063100_br_selec_1_split_0', 'LEEC36__0__20170121_080100_br_selec_3_split_0', 'LEEC09__0__20161212_051400_aa_selec_5_split_0', 'LEEC40__0__20161211_072900_aa_selec_3_split_0', 'LEEC33__0__20161115_051400_ma_selec_8_split_0', 'LEEC24__0__20161128_072900_aa_selec_3_split_0', 'LEEC23__0__20161110_054600_ma_selec_1_split_0', 'LEEC40__0__20170104_054600_br_selec_1_split_0', 'LEEC23__0__20161214_081400_aa_selec_1_split_3', 'LEEC23__0__20161214_081400_aa_selec_1_split_5', 'LEEC05__0__20161208_055900_aa_selec_2_split_0', 'LEEC36__0__20161116_081400_ma_selec_2_split_0', 'LEEC33__0__20161025_072900_ma_selec_1_split_0', 'LEEC23__0__20161125_050100_aa_selec_1_split_0', 'LEEC41__0__20161128_051400_aa_selec_3_split_0', 'LEEC18__0__20161116_051400_ma_selec_1_split_0']
'''
#DC_indices
noisy_filenames = ['LEEC02__0__20161203_063100_ma_selec_1_split_0', 'LEEC02__0__20161206_064400_ma_selec_3_split_0', 'LEEC02__0__20161210_071600_ma_selec_2_split_0', 'LEEC02__0__20170103_080100_aa_selec_1_split_0', 'LEEC03__0__20161031_055900_ma_selec_7_split_0', 'LEEC03__0__20161031_055900_ma_selec_7_split_1', 'LEEC03__0__20161111_081400_ma_selec_1_split_0', 'LEEC03__0__20161117_063100_ma_selec_5_split_0', 'LEEC03__0__20161120_051400_ma_selec_4_split_0', 'LEEC03__0__20161123_050100_aa_selec_5_split_0', 'LEEC03__0__20161123_055900_aa_selec_5_split_0', 'LEEC03__0__20161125_064400_aa_selec_5_split_0', 'LEEC03__0__20161125_072900_aa_selec_5_split_0', 'LEEC03__0__20170101_072900_br_selec_3_split_0', 'LEEC03__0__20170102_051400_br_selec_7_split_0', 'LEEC03__0__20170103_063100_br_selec_7_split_0', 'LEEC03__0__20170106_054600_br_selec_2_split_0', 'LEEC03__0__20170111_064400_br_selec_8_split_0', 'LEEC03__0__20170111_080100_br_selec_1_split_0', 'LEEC05__0__20161021_080100_ma_selec_4_split_0', 'LEEC05__0__20161030_071600_ma_selec_3_split_0', 'LEEC05__0__20161109_055900_ma_selec_3_split_0', 'LEEC05__0__20161204_055900_aa_selec_3_split_0', 'LEEC05__0__20161208_081400_aa_selec_2_split_0', 'LEEC05__0__20161218_081400_aa_selec_2_split_0', 'LEEC05__0__20170115_071600_br_selec_2_split_0', 'LEEC06__0__20161102_051400_ma_selec_3_split_0', 'LEEC06__0__20161103_050100_ma_selec_7_split_0', 'LEEC06__0__20161105_050100_ma_selec_3_split_0', 'LEEC06__0__20161107_072900_ma_selec_7_split_0', 'LEEC06__0__20161116_050100_ma_selec_3_split_0', 'LEEC06__0__20161120_051400_aa_selec_8_split_0', 'LEEC06__0__20161120_071600_aa_selec_4_split_0', 'LEEC06__0__20161121_051400_aa_selec_10_split_0', 'LEEC06__0__20161129_081400_aa_selec_2_split_0', 'LEEC06__0__20161202_071600_aa_selec_3_split_0', 'LEEC06__0__20161203_063100_aa_selec_2_split_0', 'LEEC06__0__20161209_222400_aa_selec_9_split_0', 'LEEC06__0__20161211_054600_aa_selec_4_split_0', 'LEEC06__0__20161213_054600_aa_selec_2_split_0', 'LEEC06__0__20161213_081400_aa_selec_5_split_0', 'LEEC06__0__20161218_063100_aa_selec_6_split_0', 'LEEC06__0__20161230_200100_br_selec_2_split_0', 'LEEC06__0__20170106_055900_br_selec_1_split_0', 'LEEC08__0__20161022_064400_ma_selec_4_split_0', 'LEEC08__0__20161026_081400_ma_selec_3_split_0', 'LEEC08__0__20161031_054600_ma_selec_2_split_0', 'LEEC08__0__20161103_081400_ma_selec_5_split_0', 'LEEC08__0__20161104_213900_ma_selec_3_split_0', 'LEEC08__0__20161105_055900_ma_selec_3_split_0', 'LEEC08__0__20161106_055900_ma_selec_1_split_0', 'LEEC08__0__20161107_054600_ma_selec_1_split_0', 'LEEC08__0__20161108_050100_ma_selec_2_split_0', 'LEEC08__0__20161109_200100_ma_selec_2_split_0', 'LEEC08__0__20161109_204600_ma_selec_1_split_0', 'LEEC08__0__20161113_081400_ma_selec_5_split_0', 'LEEC08__0__20161114_054600_ma_selec_4_split_0', 'LEEC08__0__20161114_072900_ma_selec_6_split_0', 'LEEC08__0__20161115_050100_ma_selec_2_split_0', 'LEEC08__0__20161116_054600_ma_selec_2_split_0', 'LEEC08__0__20161126_055900_aa_selec_4_split_0', 'LEEC08__0__20161203_204600_aa_selec_1_split_0', 'LEEC08__0__20161213_051400_aa_selec_3_split_0', 'LEEC08__0__20170104_054600_br_selec_4_split_0', 'LEEC08__0__20170117_080100_br_selec_1_split_0', 'LEEC09__0__20161117_063100_ma_selec_2_split_0', 'LEEC09__0__20161117_071600_ma_selec_7_split_0', 'LEEC09__0__20161122_050100_aa_selec_3_split_0', 'LEEC09__0__20161125_072900_aa_selec_1_split_0', 'LEEC09__0__20161127_051400_aa_selec_3_split_0', 'LEEC09__0__20161127_051400_aa_selec_4_split_0', 'LEEC09__0__20161128_050100_aa_selec_7_split_0', 'LEEC09__0__20161129_080100_aa_selec_3_split_0', 'LEEC09__0__20161204_055900_aa_selec_1_split_0', 'LEEC09__0__20161204_055900_aa_selec_4_split_0', 'LEEC09__0__20161204_064400_aa_selec_2_split_0', 'LEEC09__0__20161204_064400_aa_selec_3_split_0', 'LEEC09__0__20161209_051400_aa_selec_2_split_0', 'LEEC09__0__20161210_072900_aa_selec_7_split_0', 'LEEC09__0__20161210_080100_aa_selec_3_split_0', 'LEEC09__0__20161213_051400_aa_selec_1_split_0', 'LEEC09__0__20161213_072900_aa_selec_4_split_0', 'LEEC09__0__20161213_081400_aa_selec_3_split_0', 'LEEC09__0__20161222_080100_aa_selec_6_split_0', 'LEEC09__0__20161222_080100_aa_selec_8_split_0', 'LEEC15__0__20161020_072900_ma_selec_3_split_0', 'LEEC15__0__20161022_063100_ma_selec_5_split_0', 'LEEC15__0__20161024_063100_ma_selec_5_split_0', 'LEEC15__0__20161028_080100_ma_selec_5_split_0', 'LEEC15__0__20161029_051400_ma_selec_3_split_0', 'LEEC15__0__20161103_055900_ma_selec_5_split_0', 'LEEC15__0__20161109_080100_ma_selec_1_split_0', 'LEEC15__0__20161114_054600_ma_selec_1_split_0', 'LEEC15__0__20161124_071600_aa_selec_5_split_0', 'LEEC15__0__20161127_055900_aa_selec_3_split_0', 'LEEC15__0__20161127_081400_aa_selec_1_split_0', 'LEEC15__0__20161128_055900_aa_selec_3_split_0', 'LEEC15__0__20161210_064400_aa_selec_4_split_0', 'LEEC15__0__20161212_072900_aa_selec_1_split_0', 'LEEC15__0__20161216_063100_aa_selec_6_split_0', 'LEEC15__0__20170119_200100_br_selec_4_split_0', 'LEEC18__0__20161104_064400_ma_selec_1_split_0', 'LEEC18__0__20161123_071600_aa_selec_3_split_0', 'LEEC18__0__20161124_054600_aa_selec_3_split_0', 'LEEC18__0__20161126_054600_aa_selec_3_split_0', 'LEEC18__0__20161202_054600_aa_selec_4_split_0', 'LEEC18__0__20161202_054600_aa_selec_8_split_0', 'LEEC18__0__20161210_064400_aa_selec_5_split_0', 'LEEC18__0__20161217_081400_aa_selec_1_split_0', 'LEEC18__0__20161218_055900_aa_selec_5_split_0', 'LEEC18__0__20161221_064400_aa_selec_6_split_0', 'LEEC18__0__20170112_063100_br_selec_3_split_0', 'LEEC18__0__20170113_072900_br_selec_4_split_0', 'LEEC18__0__20170114_064400_br_selec_2_split_0', 'LEEC21__0__20161022_055900_ma_selec_4_split_0', 'LEEC21__0__20161022_063100_ma_selec_1_split_0', 'LEEC21__0__20161022_080100_ma_selec_4_split_0', 'LEEC21__0__20161024_063100_ma_selec_7_split_0', 'LEEC21__0__20161107_071600_ma_selec_4_split_0', 'LEEC21__0__20161108_055900_ma_selec_3_split_0', 'LEEC21__0__20161108_055900_ma_selec_3_split_1', 'LEEC21__0__20161119_051400_ma_selec_2_split_0', 'LEEC21__0__20161119_051400_ma_selec_5_split_0', 'LEEC21__0__20161124_050100_aa_selec_3_split_0', 'LEEC21__0__20161212_063100_aa_selec_14_split_0', 'LEEC21__0__20161216_050100_aa_selec_1_split_0', 'LEEC21__0__20161220_063100_aa_selec_4_split_0', 'LEEC21__0__20170101_081400_br_selec_3_split_0', 'LEEC21__0__20170102_063100_br_selec_3_split_0', 'LEEC21__0__20170108_080100_br_selec_2_split_0', 'LEEC23__0__20161218_054600_aa_selec_2_split_0', 'LEEC24__0__20161021_050100_ma_selec_2_split_0', 'LEEC24__0__20161120_054600_aa_selec_1_split_0', 'LEEC24__0__20161120_063100_aa_selec_7_split_0', 'LEEC24__0__20161122_071600_aa_selec_5_split_0', 'LEEC24__0__20161123_064400_aa_selec_2_split_0', 'LEEC24__0__20161125_081400_aa_selec_2_split_0', 'LEEC24__0__20161127_051400_aa_selec_5_split_0', 'LEEC24__0__20161128_072900_aa_selec_3_split_0', 'LEEC24__0__20161201_055900_aa_selec_2_split_0', 'LEEC24__0__20161202_054600_aa_selec_4_split_0', 'LEEC24__0__20161202_081400_aa_selec_7_split_0', 'LEEC24__0__20161205_072900_aa_selec_2_split_0', 'LEEC24__0__20161205_072900_aa_selec_3_split_0', 'LEEC24__0__20161207_051400_aa_selec_6_split_0', 'LEEC24__0__20161214_071600_aa_selec_6_split_0', 'LEEC24__0__20161215_063100_aa_selec_2_split_0', 'LEEC24__0__20161231_050100_br_selec_5_split_0', 'LEEC26__0__20161023_055900_ma_selec_2_split_0', 'LEEC26__0__20161107_080100_ma_selec_1_split_0', 'LEEC26__0__20161121_071600_ma_selec_2_split_0', 'LEEC26__0__20161206_055900_aa_selec_5_split_0', 'LEEC26__0__20161227_055900_aa_selec_2_split_0', 'LEEC33__0__20161026_050100_ma_selec_1_split_0', 'LEEC33__0__20161026_080100_ma_selec_5_split_0', 'LEEC33__0__20161110_071600_ma_selec_5_split_0', 'LEEC33__0__20161112_080100_ma_selec_1_split_0', 'LEEC33__0__20161214_072900_aa_selec_3_split_0', 'LEEC33__0__20170103_054600_br_selec_3_split_0', 'LEEC33__0__20170110_071600_br_selec_6_split_0', 'LEEC33__0__20170115_064400_br_selec_4_split_0', 'LEEC33__0__20170124_200100_br_selec_4_split_0', 'LEEC36__0__20161020_063100_ma_selec_6_split_0', 'LEEC36__0__20161022_051400_ma_selec_4_split_0', 'LEEC36__0__20161103_051400_ma_selec_5_split_0', 'LEEC36__0__20161109_071600_ma_selec_5_split_0', 'LEEC36__0__20161116_051400_ma_selec_2_split_0', 'LEEC36__0__20161116_054600_ma_selec_2_split_0', 'LEEC36__0__20161116_063100_ma_selec_3_split_0', 'LEEC36__0__20161116_081400_ma_selec_6_split_0', 'LEEC36__0__20161117_054600_ma_selec_4_split_0', 'LEEC36__0__20161117_054600_ma_selec_5_split_0', 'LEEC36__0__20161117_063100_ma_selec_3_split_0', 'LEEC36__0__20161117_064400_ma_selec_4_split_0', 'LEEC36__0__20161123_064400_aa_selec_2_split_0', 'LEEC36__0__20161123_064400_aa_selec_7_split_0', 'LEEC36__0__20161125_054600_aa_selec_7_split_0', 'LEEC36__0__20161127_064400_aa_selec_4_split_0', 'LEEC36__0__20161202_050100_aa_selec_4_split_0', 'LEEC36__0__20161207_063100_aa_selec_1_split_0', 'LEEC36__0__20161213_051400_aa_selec_5_split_0', 'LEEC36__0__20161214_055900_aa_selec_3_split_0', 'LEEC36__0__20161214_055900_aa_selec_7_split_0', 'LEEC36__0__20161214_221600_aa_selec_1_split_0', 'LEEC36__0__20161217_063100_aa_selec_2_split_0', 'LEEC36__0__20161219_051400_aa_selec_2_split_0', 'LEEC36__0__20161221_080100_aa_selec_4_split_0', 'LEEC36__0__20161229_050100_br_selec_9_split_0', 'LEEC36__0__20161230_221600_br_selec_2_split_0', 'LEEC36__0__20170101_080100_br_selec_8_split_0', 'LEEC36__0__20170101_081400_br_selec_4_split_0', 'LEEC36__0__20170109_054600_br_selec_7_split_0', 'LEEC37__0__20161022_055900_ma_selec_1_split_0', 'LEEC37__0__20161022_055900_ma_selec_3_split_0', 'LEEC37__0__20161026_054600_ma_selec_2_split_0', 'LEEC37__0__20161104_064400_ma_selec_1_split_0', 'LEEC37__0__20161107_081400_ma_selec_10_split_0', 'LEEC37__0__20161118_072900_ma_selec_3_split_0', 'LEEC37__0__20161120_050100_aa_selec_1_split_0', 'LEEC37__0__20161130_050100_aa_selec_3_split_0', 'LEEC40__0__20170104_055900_br_selec_1_split_0', 'LEEC41__0__20161022_063100_ma_selec_3_split_0', 'LEEC41__0__20161128_064400_aa_selec_2_split_0', 'LEEC41__0__20161201_080100_aa_selec_2_split_0', 'LEEC41__0__20161202_072900_aa_selec_4_split_0', 'LEEC41__0__20161206_064400_aa_selec_10_split_0', 'LEEC41__0__20161206_071600_aa_selec_1_split_0', 'LEEC41__0__20161209_071600_aa_selec_1_split_0', 'LEEC41__0__20161210_055900_aa_selec_4_split_0', 'LEEC41__0__20161211_071600_aa_selec_3_split_0', 'LEEC41__0__20161215_055900_aa_selec_2_split_0', 'LEEC42__0__20161023_063100_ma_selec_1_split_0', 'LEEC42__0__20161024_071600_ma_selec_1_split_0', 'LEEC42__0__20161025_071600_ma_selec_2_split_0', 'LEEC42__0__20161027_063100_ma_selec_2_split_0', 'LEEC42__0__20161028_072900_ma_selec_5_split_0', 'LEEC42__0__20161116_051400_ma_selec_2_split_0', 'LEEC42__0__20161222_064400_aa_selec_1_split_0', 'LEEC42__0__20170101_080100_br_selec_4_split_0', 'LEEC42__0__20170103_055900_br_selec_1_split_0', 'LEEC42__0__20170103_081400_br_selec_5_split_0', 'LEEC42__0__20170106_081400_br_selec_1_split_0', 'LEEC42__0__20170107_063100_br_selec_1_split_0', 'LEEC42__0__20170107_063100_br_selec_3_split_0', 'LEEC42__0__20170108_071600_br_selec_1_split_0', 'LEEC42__0__20170112_071600_br_selec_1_split_0', 'LEEC42__0__20170112_072900_br_selec_2_split_0', 'LEEC42__0__20170117_064400_br_selec_2_split_0', 'LEEC43__0__20161021_063100_ma_selec_2_split_0', 'LEEC43__0__20161101_064400_ma_selec_4_split_0', 'LEEC43__0__20161103_200100_ma_selec_6_split_0', 'LEEC43__0__20161109_063100_ma_selec_1_split_0', 'LEEC43__0__20161112_064400_ma_selec_1_split_0', 'LEEC43__0__20161112_071600_ma_selec_1_split_0', 'LEEC43__0__20161116_072900_ma_selec_9_split_0', 'LEEC43__0__20161117_071600_ma_selec_8_split_0', 'LEEC43__0__20161213_064400_aa_selec_3_split_0', 'LEEC43__0__20161218_080100_aa_selec_1_split_0', 'LEEC45__0__20161028_054600_ma_selec_4_split_0', 'LEEC45__0__20161101_050100_ma_selec_1_split_0', 'LEEC45__0__20161109_072900_ma_selec_1_split_0', 'LEEC45__0__20161109_072900_ma_selec_6_split_0', 'LEEC45__0__20161110_072900_ma_selec_2_split_0', 'LEEC45__0__20161111_051400_ma_selec_3_split_0', 'LEEC45__0__20161111_072900_ma_selec_3_split_0', 'LEEC45__0__20161215_072900_aa_selec_3_split_0', 'LEEC45__0__20161217_081400_aa_selec_3_split_0', 'LEEC45__1__20170119_221600_br_selec_2_split_1', 'LEEC48__0__20161031_054600_ma_selec_6_split_0', 'LEEC48__0__20161101_055900_ma_selec_6_split_0', 'LEEC48__0__20161108_063100_ma_selec_1_split_0', 'LEEC48__0__20161108_071600_ma_selec_6_split_0', 'LEEC48__0__20161120_081400_aa_selec_6_split_0', 'LEEC48__0__20161123_072900_aa_selec_6_split_0', 'LEEC48__0__20161208_081400_aa_selec_3_split_0', 'LEEC48__0__20161209_050100_aa_selec_5_split_0', 'LEEC48__0__20170118_064400_br_selec_3_split_0', 'LEEC48__0__20170121_055900_br_selec_2_split_0', 'LEEC49__0__20161104_081400_ma_selec_3_split_0', 'LEEC49__0__20161107_081400_ma_selec_2_split_0', 'LEEC49__0__20161112_080100_ma_selec_3_split_0', 'LEEC49__0__20161130_081400_aa_selec_4_split_0', 'LEEC49__0__20161205_071600_aa_selec_7_split_0', 'LEEC49__0__20161207_063100_aa_selec_2_split_0', 'LEEC49__0__20161207_063100_aa_selec_3_split_0', 'LEEC49__0__20161211_080100_aa_selec_10_split_0', 'LEEC49__0__20161216_050100_aa_selec_6_split_0', 'LEEC49__0__20161216_050100_aa_selec_8_split_0', 'LEEC49__0__20161216_071600_aa_selec_2_split_0', 'LEEC49__0__20161221_064400_aa_selec_5_split_0', 'LEEC49__0__20170115_081400_br_selec_1_split_0', 'LEEC49__0__20170118_054600_br_selec_1_split_0']
# noisy_indices = [item for item in noisy_indices if item not in excluded_list_index]
# df_noisy = df.loc[noisy_indices]
# df = df.drop(noisy_indices)

df_noisy = df[df['Final_filename'].isin(noisy_filenames)]
df = df[~df['Final_filename'].isin(noisy_filenames)]
# df_noisy = df_noisy.drop(df_excluded.index.tolist())
print(df_noisy)

x = df[['Final_filename', 'File_id']].values.tolist()
y = df['Species'].tolist()

x_noisy = df_noisy[['Final_filename', 'File_id']].values.tolist()
# print(x_noisy)
# print("##############################################")
y_noisy = df_noisy['Species'].tolist()
# fold_splits = stratified_Kfold_split(5, x, y, 1/8, 'folds.csv')
fold_splits = stratified_Kfold_split(5, x, y, 1/8, '../data/generated_data/folds_without_label_noise.csv')
print("##############################################")
for fold in fold_splits:
	# print(fold[5])
	# break
	create_direc(fold[1] + x_noisy, fold[2] + y_noisy, 'train', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/train_noise/' + fold[0])
	create_direc(fold[3], fold[4], 'test', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/train_noise/' + fold[0])
	create_direc(fold[5], fold[6], 'val', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/train_noise/'+ fold[0])

	create_direc(fold[1], fold[2], 'train', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/test_noise/' + fold[0])
	create_direc(fold[3]+ x_noisy, fold[4] + y_noisy, 'test', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/test_noise/' + fold[0])
	create_direc(fold[5], fold[6], 'val', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/test_noise/'+ fold[0])

	create_direc(fold[1], fold[2], 'train', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/val_noise/' + fold[0])
	create_direc(fold[3], fold[4], 'test', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/val_noise/' + fold[0])
	create_direc(fold[5]+ x_noisy, fold[6] + y_noisy, 'val', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/val_noise/'+ fold[0])

	create_direc(fold[1], fold[2], 'train', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/no_noise/' + fold[0])
	create_direc(fold[3], fold[4], 'test', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/no_noise/' + fold[0])
	create_direc(fold[5], fold[6], 'val', '/data/generated_data/transformed/none/imgs_cropped_min', '/data/generated_data/network_inputs/label_noise/no_noise/'+ fold[0])
'''
'''