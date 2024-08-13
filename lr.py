import sys
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from helper import * 
from pooling_functions import *

from tqdm import tqdm


# ------------ Functions for making plots ------------ #
def get_heatmaps(model, used_pool_func_name, save_filepath):
	weights = model.coef_

	if used_pool_func_name == "pool_concat":
		# Label feature names with their position in word
		all_features = FEATURE_NAMES
		feature_names = []
		for i in range(5):
			for feat in all_features:
				feature_names.append(f"{feat}_{i}")

		class_names = ["W AH0", "L EY0", "Y IY0"]
		weights_df = pd.DataFrame(weights, columns=feature_names,
								index=class_names).round(2)

		fig, ax = plt.subplots(figsize=(30, 8))
		sns.heatmap(weights_df, annot=False, cmap='coolwarm', center=0, cbar=False, linewidths=1, linecolor='black', square=True, cbar_kws={"shrink":0.3})
	else: 
		feature_names = FEATURE_NAMES
		class_names = ["W AH0", "L EY0", "Y IY0"]
		weights_df = pd.DataFrame(weights, columns=feature_names,
								index=class_names).round(2)

		fig, ax = plt.subplots(figsize=(10, 8))
		sns.heatmap(weights_df, annot=True, cmap='coolwarm', center=0, cbar=False, linewidths=1, linecolor='black', square=True)

	plt.title(f'Weight Matrix: {TRAINING_DATA_FOLDER} ({POOLING_FUNC_name})')
	plt.tight_layout()
	plt.savefig(save_filepath, format="jpg", dpi=300)
	print(f"Saved heatmap to path: {save_filepath}")


def plot_learning_curve(class_1_accs, class_2_accs, class_3_accs, iterations, save_filepath):
	plt.figure(figsize=(10, 6))
	
	# Plotting each accuracy list
	plt.plot(iterations, class_1_accs, label='W AH0', color="red")
	plt.plot(iterations, class_2_accs, label='L EY0', color="yellow")
	plt.plot(iterations, class_3_accs, label='Y IY0', color="blue")
	
	# Adding titles and labels
	plt.title(f'{MODEL_NAME} - {TRAINING_DATA_FOLDER} - test ({POOLING_FUNC_name})')
	plt.xlabel('Number of Batches')
	plt.ylabel('Average Proportion Correct')
	
	plt.legend()
	plt.grid(False)
	plt.savefig(save_filepath, format="jpg", dpi=300)
		

if __name__ == "__main__":	

	# ------------ Initialize file paths, resource dictionaries ------------ # 

	CONS = ["P", "B", "T", "D", "K", "G", "NG", "M", "N", "L", "F",  "V", "S", "Z", "SH", "ZH", "CH", "JH", "H"]
	VOWELS = ["IH0", "EH0", "AH0", "UH0", "IY0", "UW0", "EY0", "OW0", "IH1", "EH1", "AH1", "UH1", "IY1", "UW1", "EY1", "OW1", "IH2", "EH2", "AH2", "UH2", "IY2", "UW2", "EY2", "OW2"]

	FEATURES_FILE = "Feature_files/featsNew"
	FEATURE_NAMES = ["cons", "syll", "son", "approx", "voice", "cont", "nas", "strid", "lab", "cor", "ant", "dist", "dor", "high", "back", "tense", "diph", "stress", "main"]

	# FEATURES_FILE = "Feature_files/featsNew_mini"
	# FEATURE_NAMES = ["son", "voice", "strid"]

	symbol2feats, suffix2label, label2suffix = init_resource_dicts(FEATURES_FILE)

	# ------------ Model hyperparameters ------------ # 
	MODEL_NAME = "LR"

	TRAINING_DATA_FOLDER = "MajDefault"
	FILE_PREFIX = "majDefault"

	POOLING_FUNC = pool_last # TODO change pooling function
	POOLING_FUNC_name = "pool_last"

	# POOLING_FUNC = pool_concat # TODO change pooling function
	# POOLING_FUNC_name = "pool_concat"
	
	WRITE_RESULT_FOLDER = f"./{TRAINING_DATA_FOLDER}_{POOLING_FUNC_name}_Results_{MODEL_NAME}"
	# WRITE_RESULT_FOLDER = f"./{TRAINING_DATA_FOLDER}_{POOLING_FUNC_name}_Results_{MODEL_NAME}_featsmini"
	check_dir_exists(WRITE_RESULT_FOLDER)

	BATCH_SIZE = 10
	NUM_EPOCHS = 3
	NUM_REPEATS = 10 # Repeat LR model training 20 times for more reliable accuracy curve
	LEARNING_RATE = 0.01

	VALIDATION_FILE_PREFIX = "test"

	train_data_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_train.txt" if TRAINING_DATA_FOLDER != "MinDefault_islands" else f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_train_withIslands.txt"

	# Train classifier
	train_SGs, train_PLs, train_Ls = process_file(train_data_filepath)
	X_train, y_train = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC)
	# Copy and concatenation X_train and y_train to simulate epochs. Doing so prevents deeply nested training loops
	X_train = np.concatenate([X_train] * NUM_EPOCHS, axis=0)
	y_train = np.concatenate([y_train] * NUM_EPOCHS, axis=0)
	classes = np.unique(y_train)

	class_1_accs, class_2_accs, class_3_accs = [], [], []
	for i in tqdm(range(NUM_REPEATS)):

		run_class_1_accs, run_class_2_accs, run_class_3_accs = [], [], []

		model = SGDClassifier(loss="log_loss", max_iter=1, tol=None, warm_start=True, eta0=LEARNING_RATE, learning_rate="constant") # TODO adjust parameters of model

		curr_batch_num = 0
		for j in range(0, X_train.shape[0], BATCH_SIZE):

			X_batch = X_train[j: j+BATCH_SIZE]
			y_batch = y_train[j: j+BATCH_SIZE]
			model.partial_fit(X_batch, y_batch, classes=classes)

			test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_test.txt" # TODO change for learning curves of different conditions

			test_SGs, test_PLs, test_Ls = process_file(test_filepath)
			X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, override_max_syll=5)
			y_pred = model.predict(X_test)
			acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)

			# Save a 'snapshot' of weight heatmap every 50 batches, as well as every 10 batches for the first 50 batches
			if (i == 0) and (curr_batch_num % 50 == 0):
				heatmap_save_path = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_{VALIDATION_FILE_PREFIX}_{POOLING_FUNC_name}_HEATMAP_{curr_batch_num}.jpg"  
				get_heatmaps(model, POOLING_FUNC_name, heatmap_save_path)

				write_filepath = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_{VALIDATION_FILE_PREFIX}_{POOLING_FUNC_name}_RESULT_{curr_batch_num}.txt" 
				acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)
				write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath, suffix2label, label2suffix)

			elif (i == 0) and (curr_batch_num < 50) and (curr_batch_num % 10 == 0):
				heatmap_save_path = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_{VALIDATION_FILE_PREFIX}_{POOLING_FUNC_name}_HEATMAP_{curr_batch_num}.jpg"  
				get_heatmaps(model, POOLING_FUNC_name, heatmap_save_path)

				write_filepath = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_{VALIDATION_FILE_PREFIX}_{POOLING_FUNC_name}_RESULT_{curr_batch_num}.txt" 
				acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)
				write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath, suffix2label, label2suffix)

			run_class_1_accs.append(acc_dict["W AH0"])
			run_class_2_accs.append(acc_dict["L EY0"])
			run_class_3_accs.append(acc_dict["Y IY0"])

			curr_batch_num += 1

		class_1_accs.append(run_class_1_accs)
		class_2_accs.append(run_class_2_accs)
		class_3_accs.append(run_class_3_accs)

	# Take average of accs across all runs
	avg_class_1_accs = [sum(x) / len(x) for x in zip(*class_1_accs)]
	avg_class_2_accs = [sum(x) / len(x) for x in zip(*class_2_accs)]
	avg_class_3_accs = [sum(x) / len(x) for x in zip(*class_3_accs)]

	# Plot learning curve
	curve_save_path = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_{VALIDATION_FILE_PREFIX}_{POOLING_FUNC_name}_CURVE.jpg" 
	num_batches = int(np.ceil(X_train.shape[0]/BATCH_SIZE))
	iterations = [i for i in range(num_batches)]
	plot_learning_curve(avg_class_1_accs, avg_class_2_accs, avg_class_3_accs, iterations, curve_save_path)

	# Plot weight heatmap
	heatmap_save_path = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_{VALIDATION_FILE_PREFIX}_{POOLING_FUNC_name}_HEATMAP_FINAL.jpg"  
	get_heatmaps(model, POOLING_FUNC_name, heatmap_save_path)
	
	test_file_suffixes = ["test", "test_H", "test_L", "test_Mutants", "testNewTemplates"]
	for test_file_suffix in test_file_suffixes:
		print(f"Testing {test_file_suffix}")
		test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_{test_file_suffix}.txt"
		write_filepath = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_{test_file_suffix}_{POOLING_FUNC_name}_RESULT.txt" 

		test_SGs, test_PLs, test_Ls = process_file(test_filepath)
		X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, override_max_syll=5)
		y_pred = model.predict(X_test)
		acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)

		print(f'{acc_dict["W AH0"]} - {acc_dict["L EY0"]} - {acc_dict["Y IY0"]}')

		write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath, suffix2label, label2suffix)
