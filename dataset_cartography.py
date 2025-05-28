import pandas as pd
import numpy as np
import re
from scipy import signal
from dataset_cartography_gbdt import dataset_cartography, plot_dataset_cartography
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
import shap
import os
import ast
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42

def normalize_dataframe(df):
	# Check if the DataFrame has numerical columns
	numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
	
	# Normalize the numerical columns
	df_normalized = df.copy()
	for column in numerical_columns:
		min_val = df_normalized[column].min()
		max_val = df_normalized[column].max()
		
		# Apply min-max normalization (between 0 and 1)
		df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
	
	return df_normalized


def extract_time_of_day(df):
	# Get the filename column 
	print(df)
	filename_column = df.loc[:, 'filename']
	print(filename_column)
	# Define the pattern: underscore, 6 digits, followed by another underscore
	pattern = r'_(\d{6})_' 
	
	# Extract the first 2 digits of the 6 digits from the matching pattern
	result = filename_column.apply(lambda x: re.search(pattern, str(x)) and re.search(pattern, str(x)).group(1)[:2] if isinstance(x, str) else None)
	arr_time = []
	for value in result:
		# print(int(value))
		if(int(value) < 12):
			arr_time.append(0)
		else:
			arr_time.append(1)
	# print(arr_time)
	return arr_time

def extract_location(df):
	# Get the filename column info
	filename_column = df.loc[:, 'filename']
	 # Define the pattern: underscore, 2 letters, followed by another underscore
	pattern = r'_([a-z]{2})_'
	result = filename_column.str.findall(pattern)
	print(result)

	arr_loc = []
	for value in result:

	  arr_loc.append(value[0])
	return arr_loc

def remove_correlated_columns(df, threshold=0.9):
	# Select specific columns from the dataframe
	df = df.iloc[:, np.r_[1:37]]
	# print(df)
	df = normalize_dataframe(df)
	# print(df)
	
	# Calculate the correlation matrix between the columns
	correlation = df.corr()
	# print(correlation)
	
	# Create a list of columns to remove
	columns_to_remove = []
	
	# Iterate over the columns to check for correlations
	for i in range(len(correlation.columns)):
		for j in range(i):
			if abs(correlation.iloc[i, j]) > threshold:
				# If the correlation between columns is higher than the threshold, remove the column with lower variance
				col1 = correlation.columns[i]
				col2 = correlation.columns[j]
				# print(col1, col2)
				
				variance_col1 = df[col1].var()
				variance_col2 = df[col2].var()
				# print(variance_col1, variance_col2)
				
				if variance_col1 > variance_col2:
					columns_to_remove.append(col2)
					# print('removed: ', col2)
				else:
					columns_to_remove.append(col1)
					# print('removed: ', col1)
	
	# Drop the columns that are in the list
	df_clean = df.drop(columns=columns_to_remove)
	
	return df_clean, columns_to_remove


def prepare_dataframe_DC(df, geo_location):
	df_normalized = normalize_dataframe(df)
	df_clean_correlated, _ = remove_correlated_columns(df_normalized)
	df_clean_correlated['is_PM'] = extract_time_of_day(df_clean_correlated)
	if(geo_location == 'True'):
		df_clean_correlated['aa'] = 0
		df_clean_correlated['br'] = 0
		df_clean_correlated['ma'] = 0
		loc = extract_location(df_clean_correlated)
		for i, location in enumerate(loc):
			# print(i, location)
			df_clean_correlated.loc[i, location] = 1
		# print(df[['original_filename', 'aa', 'br', 'ma', 'is_PM']])
	return df_clean_correlated

from scipy.signal import argrelextrema


def calculate_derivative_y_only(y):
	"""
	Calculate the numerical derivative of a set of y values, assuming equally spaced x values.

	Parameters:
	- y: List or array of y-values (dependent variable).

	Returns:
	- derivative: List or array of the derivative values (dy/dx).
	"""
	
	# Convert y to a numpy array for easier calculation
	y = np.asarray(y)
	
	# Derivative for equally spaced x-values (assumed to be spaced by 1)
	derivative = np.diff(y)  # Difference of y-values (dy/dx)
	
	# Optionally, you can append a NaN to the derivative to match the original length
	# since the derivative reduces the number of points by one.
	derivative = np.append(derivative, np.nan)  # Append NaN to match length
	
	return derivative

def check_avg_surrounding_greater_specified(y, indices, window_size=5):
	"""
	Check for specific points whether the average of the previous and next `window_size` values
	is greater or smaller than the current value at those points.

	Parameters:
	- y: List or array of y-values (dependent variable).
	- indices: List of indices at which to check the surrounding values.
	- window_size: The number of surrounding values to consider (default is 5 for both sides).
	
	Returns:
	- None (Prints the result for each specified index).
	"""
	n = len(y)
	
	for i in indices:
		# Make sure the index is within bounds
		if i < 0 or i >= n:
			print(f"Index {i} is out of bounds!")
			continue
		
		# Determine the range of previous and next values to include in the average
		prev_start = max(i - window_size, 0)  # Ensure we don't go out of bounds
		prev_end = i  # Not including the current point
		next_start = i + 1  # Not including the current point
		next_end = min(i + window_size + 1, n)  # Ensure we don't go out of bounds
		
		# Calculate the previous and next values
		prev_values = y[prev_start:prev_end]
		next_values = y[next_start:next_end]
		
		# Combine the values from previous and next
		surrounding_values = prev_values + next_values
		
		# Only calculate the average if there are surrounding values
		if len(surrounding_values) > 0:
			avg_surrounding = np.mean(surrounding_values)
			# Output whether the average of surrounding values is greater or smaller than the current value
			comparison = "greater" if avg_surrounding > y[i] else "smaller"
			print(f"Index {i}: Average of surrounding values is {comparison} than current value ({y[i]}).")
		else:
			print(f"Index {i}: Not enough surrounding data.")



def find_valley_points(data, plot=False, order=5):
	"""
	Find the valley points (local minima) in a violin plot and return values below those valleys.

	Parameters:
	- data: Array-like, the data to analyze (1D array or list).
	- plot: Boolean, if True, a plot with valley points will be displayed.
	- order: Integer, the size of the window for finding local minima (larger values increase sensitivity).

	Returns:
	- valley_points: List of x-values where valleys occur in the density curve.
	- values_below_valley: List of data values that are below the valley points.
	"""
	# Ensure data is a numpy array for easier comparison
	data = np.asarray(data)
	
	# Create the violin plot to extract data
	sns.violinplot(data=data)
	
	# Get the current axis of the plot
	ax = plt.gca()

	# Extract the data used by the violinplot (this includes the kernel density estimation)
	paths = ax.collections[0].get_paths()

	# Assume the first path corresponds to the density plot (usually for 1D data)
	path = paths[0]

	# Extract the x and y coordinates (density values)
	x = path.vertices[:, 0]  # x values (quantiles of the data)
	y = path.vertices[:, 1]  # y values (density)
	print(len(x))
	print(x)
	print('------------')
	print(y)
	print('------------')
	print(max(data))
	print(min(data))
	print('------------')
	new_x = x[:int(len(x)/2)]
	# print(x)
	# print(y)
	# print(check_avg_surrounding_greater(x))
	# print(calculate_derivative_y_only(x))

	# Find the valley points: local minima in the y-values (density)
	valleys_indices = argrelextrema(new_x, np.greater, order=order)[0]
	print("valleys_indices:", valleys_indices)
	print('values:', x[valleys_indices])
	print(check_avg_surrounding_greater_specified(new_x, valleys_indices))
	# If no valleys are found, use a fallback approach
	if len(valleys_indices) == 0:
		print("No valleys found using argrelextrema. Trying fallback method.")
		# Fallback: Find valleys by looking for small drops in density
		valleys_indices = np.where(np.diff(x) < 0)[0]  # Find where density decreases
	# valleys_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
	# The valley points (local minima) are the x-values corresponding to these indices
	valley_points = y[valleys_indices[0]]
	print(valley_points)
	# Extract the values below each valley point
	values_below_valley = []
	indices_below_valley = []
	for i, value in enumerate(data):
		if value < valley_points:
			values_below_valley.append(value)
			indices_below_valley.append(i)
	print(len(values_below_valley))
	# Convert the list of values to a Pandas Series (or DataFrame if you prefer)
	# values_below_valley = pd.Series(values_below_valley)

	# Optionally plot the density curve with valley points
	if plot:
		plt.plot(x, y, label="Density curve")
		plt.scatter(x[valleys_indices], y[valleys_indices], color='red', label="Valley points", zorder=5)
		plt.legend()
		plt.savefig("violin_plot_indices.png", dpi=300, bbox_inches='tight')  # Saves the figure as a PNG file
		plt.close()

	return values_below_valley, indices_below_valley

def find_anomalies(data, labels, label_limits=10, threshold_std=2):
	"""
	Find anomalies in the data, while considering the label of each instance and limiting the number of selected anomalies per label after detection.
	The subset of anomalies selected will correspond to the smallest values of the data. 
	The limit is defined as a percentage of the total anomalies detected for each label.

	Parameters:
	- data: The dataset (a list or numpy array of data points).
	- labels: An array (or list) of labels corresponding to each data point in `data`.
	- label_limits: An array (or list) with the percentage of anomalies to be selected for each label (default is 10%).
	- threshold_std: The number of standard deviations below the mean to consider as an anomaly.

	Returns:
	- A dictionary where keys are the labels and values are the indices of anomalous data points for that label.
	"""

	# # Dictionary to store anomalies by label
	anomalies_by_label = {label: [] for label in np.unique(labels)}

	# # Perform anomaly detection on the entire dataset
	# frequency_dist, bin_edges = np.histogram(data, bins=100)
	# reverse_freq = 1 / frequency_dist

	# # Find peaks in the reverse frequency distribution
	# peaks = signal.find_peaks(reverse_freq)	
	# print("PEAKS: ", peaks)
	# print(frequency_dist)
	# print(bin_edges)
	# print(reverse_freq)
	# breakpoint()
	# peaks = list(peaks[0])

	# if len(peaks) > 0:
	# 	# If peaks are found, use the first peak to determine the new threshold
	# 	new_threshold = bin_edges[int(peaks[0])]
	# 	anomalous_indices = np.where(data < new_threshold)[0]  # Indices of anomalous data points
	# else:
	# 	# If no peaks, use the standard deviation method
	# 	avg = np.average(data)
	# 	std = np.std(data)

	# 	# Anomalies are those data points below (avg - threshold_std * std)
	# 	new_threshold = avg - (threshold_std * std)
	# 	anomalous_indices = np.where(data < new_threshold)[0]  # Indices of anomalous data points
	anomalous_values, anomalous_indices = find_valley_points(data, plot=True)
	# print(anomalous_values)
	print(anomalous_indices)
	# anomalous_indices = anomalous_indices.to_list()
	# After detecting anomalies, limit the number of anomalies per label
	print('labels:', labels)
	# labels = labels.to_list()
	print('labels:', labels)
	for label in np.unique(labels):
		# Get the indices for the current label's anomalies

		print("................................................")
		# print(label)
		# print(labels[labels == label])
		# print(labels.iloc[:, 0] == label)
		label_indices = labels[labels == label].index.to_list()
		print(label_indices)
		print("................................................")

		label_indices_anomalous = [value for value in anomalous_indices if value in label_indices]
		print(label_indices_anomalous)
		# breakpoint()
		# # Sort anomalies for the current label by the values in `data` (ascending order)
		# sorted_label_indices = sorted(label_indices, key=lambda idx: data[idx])

		# # Calculate the number of anomalies to select based on the percentage (label_limits)
		# num_anomalies = len(sorted_label_indices)
		# limit = int(np.ceil(num_anomalies * (label_limits / 100)))  # Calculate the limit based on the percentage
		# limited_label_indices = sorted_label_indices[:limit]  # Limit the number of anomalies
		limit = int(len(label_indices) * label_limits/100)
		if(len(label_indices_anomalous) > len(label_indices) * label_limits/100):
			print('More values for species ', label, 'than the limit of ', label_limits, '%')
			aux = [data[i] for i in label_indices_anomalous]
			sorted_pairs = sorted(zip(aux, label_indices_anomalous))
			aux, label_indices_anomalous = zip(*sorted_pairs)
			print(label_indices_anomalous)

			# Store the limited anomalies for each label
			anomalies_by_label[label] = label_indices_anomalous[:limit]
			# breakpoint()
		else:
			anomalies_by_label[label] = label_indices_anomalous
	return anomalies_by_label

def calculate_metrics_from_prediciton(y_true,y_pred_prob, threshold = 0.5):
	y_pred = (y_pred_prob > threshold).astype(int)
	
	mcc = metrics.matthews_corrcoef(y_true, y_pred)
	precision, recall, thresholds = metrics.precision_recall_curve(y_true,y_pred_prob)
	pr_auc = metrics.auc(recall, precision)
	precision = metrics.precision_score(y_true,y_pred,zero_division=0)
	recall = metrics.recall_score(y_true,y_pred,zero_division=0)
	f1 = metrics.f1_score(y_true,y_pred,zero_division=0)
	classification_report = metrics.classification_report(y_true, y_pred,zero_division=0)
	
	return {'mcc':mcc , 'pr_auc': pr_auc, 'f1':f1, 'precision':precision, 'recall':recall, 'classification_report':classification_report,'threshold':threshold}

def calculate_metrics(clf, X,y, threshold = 0.5):
	from sklearn import metrics
	
	y_pred_prob = clf.predict_proba(X)[:,1]
	y_pred = (y_pred_prob > threshold).astype(int)
	
	mcc = metrics.matthews_corrcoef(y, y_pred)
	# precision, recall, thresholds = metrics.precision_recall_curve(y,y_pred_prob)
	# pr_auc = metrics.auc(recall, precision)
	precision = metrics.precision_score(y,y_pred,zero_division=0, average='macro')
	recall = metrics.recall_score(y,y_pred,zero_division=0, average='macro')
	f1 = metrics.f1_score(y,y_pred,zero_division=0, average='macro')
	classification_report = metrics.classification_report(y, y_pred,zero_division=0)
	
	# return {'mcc':mcc , 'pr_auc': pr_auc, 'f1':f1, 'precision':precision, 'recall':recall, 'classification_report':classification_report,'threshold':threshold}
	return {'mcc':mcc , 'f1':f1, 'precision':precision, 'recall':recall, 'classification_report':classification_report,'threshold':threshold}

def run_experiment(X_train, y_train, EPOCHS = 100):#, X_test,y_test,EPOCHS = 100):
	
	# initial weights initialization
	weights = np.ones_like(y_train)
	# Noise reference
	# y_true_noise = (df.loc[X_train.index,"tag"] == 'label_noise').astype(int)
	# weight correction steps
	step = 1/EPOCHS
	
	#result container
	result = dict(
				weights_list = [],
				confidence_list = [],
				correctness_list = [],
				variability_list = [],
				clf_list = [],
				metrics_list = [],
				noise_metrics_list = [],
				# tag = df.loc[X_train.index,"tag"]
				)
	
	# weight correction iteration
	for i in range(EPOCHS):
		
		#fit using weights
		clf = lgbm.LGBMClassifier(random_state=SEED, is_unbalance=True, max_depth=5, num_leaves=12)
		clf.fit(X_train, y_train, sample_weight=weights)
		
		confid, variab, correc = dataset_cartography(clf, X_train, y_train,range_estimators=[10,100])
		

		result["variability_list"].append(variab)
		result["confidence_list"].append(confid)
		result["correctness_list"].append(correc)
		result["clf_list"].append(clf)
		# mask = df.loc[X_test.index,'tag']!= 'label_noise'
		# result["metrics_list"].append(calculate_metrics(clf,X_test,y_test))

		y_pred=clf.predict(X_train)
		print('LightGBM Model accuracy score train: {0:0.4f}'.format(accuracy_score(y_train, y_pred)))
		# y_pred=clf.predict(X_test)
		# print('LightGBM Model accuracy score test: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
		# print(np.max(correc))
		# print(clf.booster_.dump_model())
		print(clf.booster_.trees_to_dataframe())

		#Noise metrics
		# y_pred_proba = 1 - weights
		# result["noise_metrics_list"].append(calculate_metrics_from_prediciton(y_true_noise,y_pred_proba,threshold=0.5))
		
		#weigts correction
		weights  = np.clip(weights - step*(1 - correc*confid), 0,1)
		result["weights_list"].append(weights)		
	return result.copy()

def generate_DC_indices(df, result, X_train, METRIC_NAME = 'f1'):

	temp_df = df.loc[X_train.index,:].copy()
	temp_df['w'] =result['weights_list'][-1]
	# temp_df['threshold'] = np.array(result['confidence_list'][-1]) * np.array((result['correctness_list'][-1]).T)

	
	# ind_thresh = find_anomalies(temp_df['threshold'], temp_df['Species'])
	# print(len(ind_thresh))
	# print(ind_thresh)
	# flat_list = [num for sublist in ind_thresh.values() for num in sublist]
	# print(temp_df['threshold'])

	# from scipy.stats import gaussian_kde

	# # Fit a Gaussian KDE to the data
	# kde = gaussian_kde(temp_df['threshold'], bw_method='scott')  # bandwidth method used by default in Seaborn
	# x = np.linspace(-4, 4, 1000)
	# y = kde(x)

	# # Plot the KDE
	# plt.close()
	# plt.plot(x, y, label="KDE")
	# plt.fill_between(x, 0, y, alpha=0.5)
	# plt.title("Kernel Density Estimate")
	# plt.show()

	# plt.close()
	# plt.figure(figsize=(25, 5))
	# ax = sns.violinplot(data=temp_df['threshold'], inner='point')
	# from matplotlib import collections

	# violins = ax.collections[::2]  # Violins are the 0th, 2nd, 4th... collection items
	# for i, violin in enumerate(violins):
	# 	# Extract the path of the violin (this contains the x and y values of the curve)
	# 	path = violin.get_paths()[0]
		
	# 	# Extract x and y coordinates of the curve (density values)
	# 	x_vals = path.vertices[:, 0]
	# 	y_vals = path.vertices[:, 1]
		
	# 	# Annotate each density value on the plot
	# 	for x, y in zip(x_vals, y_vals):
	# 		ax.text(x, y, f'{y:.2f}', fontsize=8, color='black', ha='center', va='center')
	# plt.show()
	# plt.figure(figsize=(25, 5))
	# ax = sns.violinplot(data=temp_df['threshold'].drop(flat_list), inner='point')
	# plt.show()

	ind_w = find_anomalies(temp_df['w'], temp_df['Species'])
	print('@@@@@@@@@@@@@@@@@@@@@')
	print(ind_w)
	# breakpoint()
	# print(ind_thresh)
	print('@@@@@@@@@@@@@@@@@@@@@')
	# set1 = set(ind_w)
	# set2 = set(ind_thresh)
	# # Find intersection of the sets (common elements)
	# common_elements = set1.intersection(set2)
	# print(common_elements)
	# print(len(common_elements))
	# intersection_indices = {}
	# union_indices = {}
	# for key in ind_thresh.keys():
	# 	# Find the intersection of values (lists) for each key
	# 	intersection = list(set(ind_thresh.get(key, [])) & set(ind_w.get(key, [])))
	# 	union = list(set(ind_thresh.get(key, [])) | set(ind_w.get(key, [])))
	# 	# If there are common indices, store them
	# 	if intersection:
	# 		intersection_indices[key] = intersection
	# 	if union:
	# 		union_indices[key] = union

	# # Print out the common indices for each key
	# print(intersection_indices)
	# print(union_indices)

	return ind_w



def generate_DC_indices_vit(path, file_start_pred, file_start_index, file_start_label, file_start_prob, df):
	map_correc = {}
	map_conf = {}
	for dirpath, dirnames, filenames in os.walk(path):
		for filename in filenames:
			print(filename)
			if filename.startswith(file_start_pred):
				print('#####################')
				print(dirpath)
				print(filename)
				f_pred, f_index, f_label = [], [], []
				log_id = filename[len(file_start_pred):]
				with open(os.path.join(dirpath, filename), 'r') as f:
					f_pred = ast.literal_eval(f.read())
				with open(os.path.join(dirpath, file_start_index + log_id), 'r') as f:
					f_index = ast.literal_eval(f.read())[0]
				with open(os.path.join(dirpath, file_start_label + log_id), 'r') as f:
					f_label = ast.literal_eval(f.read())[0]
				with open(os.path.join(dirpath, file_start_prob + log_id), 'r') as f:
					f_prob = ast.literal_eval(f.read())
				# print('--------------')
				# print(f_pred)
				# print('--------------')
				# print(f_prob)
				# print('--------------')
				# print(f_index)
				# print('--------------')
				# print(f_label)
				# print('--------------')
				# print(len(f_pred))
				# print(len(f_pred[0]))
				# print('--------------')
				# print(len(f_prob))
				# print(len(f_prob[0]))
				# print(len(f_prob[0][0]))
				# print('--------------')
				# print(len(f_index))
				# print(len(f_index[0]))
				# print('--------------')
				# print(len(f_label))
				# print(len(f_label[0]))
				for i, epoch in enumerate(f_pred):
					for j, value in enumerate(epoch):
						key = f_index[j]
						print(key)
						if key not in map_correc:
							map_correc[key] = []
							map_conf[key] = []
						if(value == f_label[j]):
							map_correc[key].append(1)
						else:
							map_correc[key].append(0)
						map_conf[key].append(f_prob[i][j][f_label[j]])
	print(map_correc)
	print(map_conf)
				# break
	matx = [] # HARD-CODED
	matx_conf = []
	matx_correc = []
	labels = []
	ids = []

	# df = pd.read_csv('labels_LEEC.csv')

	print(len(matx))
	for key in map_correc:
		print(key)
		ids.append(key)
		tmp_ind = df.index[df['File_id'] == key].tolist()[0]
		labels.append(df.loc[tmp_ind, 'Species'])

		conf = np.mean(map_conf[key])
		correc = np.sum(map_correc[key])/len(map_correc[key])
		print(conf, correc)
		matx_conf.append(conf)
		matx_correc.append(correc)
		matx.append(correc * conf)
	print(matx)
	print(labels)
	print(ids)

	# x_values =  list(matx_correc)
	# y_values =  list(matx_conf)
	# classes =  list(labels)
	# # Identify unique classes
	# unique_classes = sorted(set(classes))
	# n_classes = len(unique_classes)

	# # Set subplot grid size (e.g., 3 rows × 4 columns)
	# ncols = 4
	# nrows = (n_classes + ncols - 1) // ncols  # ceiling division

	# # Create figure and axes
	# fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
	# axs = axs.ravel()  # Flatten to 1D for easy iteration

	# # Plot each class
	# for i, cls in enumerate(unique_classes):
	#     idx = [j for j, c in enumerate(classes) if c == cls]
	#     axs[i].scatter(
	#         [x_values[j] for j in idx],
	#         [y_values[j] for j in idx],
	#         s=30,
	#         alpha=0.7
	#     )
	#     axs[i].set_title(cls)
	#     axs[i].set_xlim(min(x_values), max(x_values))
	#     axs[i].set_ylim(min(y_values), max(y_values))
	#     axs[i].grid(True)

	# # Hide any unused subplots
	# for j in range(len(unique_classes), len(axs)):
	#     axs[j].set_visible(False)

	# # Global layout
	# # fig.suptitle('Scatter Plots by Class', fontsize=16)
	# fig.text(0.5, 0.04, 'Correctness', ha='center')
	# fig.text(0.04, 0.5, 'Confidence', va='center', rotation='vertical')

	# plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # leave room for suptitle and axis labels
	# plt.savefig('scatter_plot_conf_correc_vit.png', dpi=300)
	# plt.show()
	# breakpoint()

	classes = df['Species'].unique().tolist()
	# classes = classes#[0:7]
	dic_classes = {}

	for i, item in enumerate(classes):
		dic_classes[item] = i

	labels = pd.Series(labels).map(dic_classes)

	indices = find_anomalies(matx, labels)

	# fig ,axes = plt.subplots(ncols=1, figsize=(25,5))
	# sns.violinplot(data=matx,ax=axes)
	# plt.show()
		# break
	removed_indexes  = [num for sublist in indices.values() for num in sublist]
	print(removed_indexes)
	removed_filenames = df['Final_filename'].loc[df.index.isin(removed_indexes)].tolist()
	print("removed filenames: ", removed_filenames)
	'''
	'''
	return removed_indexes, removed_filenames


				


def get_label_noise_indexes(path_features, features_filename, df_labels, threshold = 0.9):
	# caminho_csv = '.\\vis\\indices\\vocal_vmd.csv'  # Substitua pelo caminho do seu arquivo CSV
	caminho_csv = os.path.join(path_features, features_filename)
	df = pd.read_csv(caminho_csv)
	# print(df)

	df['filename'] = df['filename'].str.replace(r'\.wav$', '', regex=True)
	labels = df_labels.loc[:, ['Species', 'Final_filename']]
	# print(labels[30:40])

	df = pd.merge(df, labels, left_on=['filename'], right_on=['Final_filename'], how='left').drop('Final_filename', axis=1)
	# print(df[30:40])
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

	print(df)

	print(df.columns)
	y_train = df['Species']
	x_train = df.drop(['Species', 'filename'], axis=1)


	model = lgbm.LGBMClassifier(random_state=SEED, is_unbalance=True, max_depth=5, num_leaves=12)
	model.fit(x_train, y_train)
	print('Species: ', y_train)

	explainer = shap.Explainer(model)
	shap_values = explainer(x_train)
	print(shap_values)
	# Visualize
	plt.figure()
	shap.summary_plot(shap_values, x_train, class_names= model.classes_, show=False)
	plt.xlabel("Mean absolut SHAP value")
	plt.savefig("shap_summary_plot.png", bbox_inches='tight')
	plt.close()
	# Mean(|SHAP|) over both samples and classes
	importance_values = np.abs(shap_values.values).mean(axis=0).mean(axis=1)

	shap_importance = pd.DataFrame({
		'feature': x_train.columns,
		'importance': importance_values
	})

	# Top 20
	top20 = shap_importance.sort_values(by='importance', ascending=False).head(20)
	print(top20)
	top20_features = top20['feature'].tolist()
	print(top20_features)

	x_train = x_train[top20_features]
	# print(x_train)
	# breakpoint()

	# _, colunas_removidas = remove_correlated_columns(df.drop('Species', axis=1), threshold)


	# # # Exibir as colunas removidas e o novo DataFrame
	# # print("Colunas removidas:", colunas_removidas)
	# # print("\nNovo DataFrame após remoção das colunas correlacionadas:")
	# # print(df)
	# df = df.drop(colunas_removidas, axis=1)
	# print(df)
	# df.to_csv('indices_normalizados.csv', index=False)
	# df.to_csv('indices_normalizados_loc_time.csv', index=False)

	#######################################

	classes = df['Species'].unique().tolist()
	# classes = classes#[0:7]
	dic_classes = {}

	for i, item in enumerate(classes):
		dic_classes[item] = i
	# print(dic_classes)

	# print(df)
	# print('########################')
	# print(df['Species'].isin(classes)[30:40])
	# print('########################')
	# df = df[df['Species'].isin(classes)]
	# wrong_species_df = df[~df['Species'].isin(classes)]
	# print('wrong_species_df: \n', wrong_species_df)
	# print(df)
	# print(df['Species'].value_counts())

	# y_train = df.loc[:,"Species"]
	# x_train = df.drop(columns = ['filename', 'Species'])
	print('x_train: ', x_train)
	print(y_train)
	classes = y_train
	y_train = y_train.map(dic_classes)
	# y_test = y_test.map(dic_classes)
	print(y_train)
	print(y_train.value_counts())

	result = run_experiment(x_train, y_train, EPOCHS = 20)

	# plot_dataset_cartography(result['confidence_list'][-1], result['variability_list'][-1], result['correctness_list'][-1], y_train, dic_classes, 'indices')
	# breakpoint()

	x_values =  list(result['correctness_list'][-1])
	y_values =  list(result['confidence_list'][-1])
	classes =  list(classes)
	# Identify unique classes
	unique_classes = sorted(set(classes))
	n_classes = len(unique_classes)

	# Set subplot grid size (e.g., 3 rows × 4 columns)
	ncols = 4
	nrows = (n_classes + ncols - 1) // ncols  # ceiling division

	# Create figure and axes
	fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
	axs = axs.ravel()  # Flatten to 1D for easy iteration

	# Plot each class
	for i, cls in enumerate(unique_classes):
	    idx = [j for j, c in enumerate(classes) if c == cls]
	    axs[i].scatter(
	        [x_values[j] for j in idx],
	        [y_values[j] for j in idx],
	        s=30,
	        alpha=0.7
	    )
	    axs[i].set_title(cls)
	    axs[i].set_xlim(min(x_values), max(x_values))
	    axs[i].set_ylim(min(y_values), max(y_values))
	    axs[i].grid(True)

	# Hide any unused subplots
	for j in range(len(unique_classes), len(axs)):
	    axs[j].set_visible(False)

	# Global layout
	# fig.suptitle('Scatter Plots by Class', fontsize=16)
	fig.text(0.5, 0.04, 'Correctness', ha='center')
	fig.text(0.04, 0.5, 'Confidence', va='center', rotation='vertical')

	plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # leave room for suptitle and axis labels
	plt.savefig('scatter_plot_conf_correc_indices.png', dpi=300)
	plt.show()
	breakpoint()

	indices = generate_DC_indices(df, result, x_train, METRIC_NAME = 'f1')
	removed_indexes  = [num for sublist in indices.values() for num in sublist]
	print(removed_indexes)
	removed_filenames = df['filename'].loc[df.index.isin(removed_indexes)]
	print("removed filenames: ", removed_filenames.to_list())

	return removed_indexes, removed_filenames

df = pd.read_csv('labels_LEEC.csv')
# print(find_valley_points(data, plot=False))
# print(find_anomalies(data, df['Species']))

get_label_noise_indexes('.\\vis\\final', 'signal_cropped_vocal_Indices.csv', df)
path  = 'C:\\Users\\GustavoLopes\\Documents\\Codigo\\Gustavo\\final_code\\final_code\\results\\final\\vit\\none\\imgs_cropped_min'
generate_DC_indices_vit(path, 'log_info_gustavo_pred_', 'log_info_gustavo_ids_', 'log_info_gustavo_labels_', 'log_info_gustavo_probs_', df)