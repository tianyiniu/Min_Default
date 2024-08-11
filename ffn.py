import sys
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from helper import * 
from pooling_functions import *


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
		sns.heatmap(weights_df, annot=False, cmap='coolwarm', center=0, cbar=True, linewidths=1, linecolor='black', square=True, cbar_kws={"shrink":0.3})
	else: 
		feature_names = FEATURE_NAMES
		class_names = ["W AH0", "L EY0", "Y IY0"]
		weights_df = pd.DataFrame(weights, columns=feature_names,
								index=class_names).round(2)

		fig, ax = plt.subplots(figsize=(10, 8))
		sns.heatmap(weights_df, annot=True, cmap='coolwarm', center=0, cbar=True, linewidths=1, linecolor='black', square=True)

	plt.title(f'{TRAINING_DATA_FOLDER}_LR_{POOLING_FUNC_name}')
	plt.tight_layout()
	plt.savefig(save_filepath, format="jpg", dpi=300)
	print(f"Saved heatmap to path: {save_filepath}")


def plot_learning_curve(class_1_accs, class_2_accs, class_3_accs, iterations):
	plt.figure(figsize=(10, 6))
	
	# Plotting each accuracy list
	plt.plot(iterations, class_1_accs, label='W AH0', color="red")
	plt.plot(iterations, class_2_accs, label='L EY0', color="yellow")
	plt.plot(iterations, class_3_accs, label='Y IY0', color="blue")
	
	# Adding titles and labels
	plt.title(f'Learning Curve - LR - Equal Default - EqualFreq - {POOLING_FUNC_name}')
	plt.xlabel('Num batches (MAX_ITERATIONS / BATCH_SIZE)')
	plt.ylabel('Accuracy')
	
	plt.legend()
	plt.grid(False)
	save_filepath = "learning_curve_temp.jpg"
	plt.savefig(save_filepath, format="jpg", dpi=300)


# ------------ Test Accuracy by Gold Label ------------ #
def calc_results_by_gold_label(y_test, y_pred):
	gold_classes = suffix2label.keys()
	acc_dict = {gold_class:[] for gold_class in gold_classes}
	
	# Calculate accuracies for each gold label
	for y_gold, y_guess in zip(y_test, y_pred):
		gold_suffix = label2suffix[y_gold]
		pred_suffix = label2suffix[y_guess]
		is_correct = 1 if gold_suffix == pred_suffix else 0
		acc_dict[gold_suffix].append(is_correct)

	for gold_class in acc_dict:
		accs_list = acc_dict[gold_class]
		if len(accs_list):
			accs = sum(accs_list)/len(accs_list)
		else:
			accs = -1
		acc = round(accs, 3)
		acc_dict[gold_class] = acc
	return acc_dict
		

	# ------------ Test Accuracy by Word Type ------------ #
def write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath):
	# Find unique word type (templates) in dataset
	word_types = set()
	for sg in test_SGs:
		segments = sg.split(" ")
		word_type = ["c" if seg in CONS else "v" for seg in segments]
		word_types.add("".join(word_type))

	# Initialize acc dict and error dict. Both indexed by word type then gold class
	gold_classes = suffix2label.keys()
	acc_dict = {word_type: {gold_class:[] for gold_class in gold_classes} for word_type in word_types}
	error_dict = {word_type: {gold_class:[] for gold_class in gold_classes} for word_type in word_types}

	for word, y_gold, y_guess in zip(test_SGs, y_test, y_pred):
		gold_suffix = label2suffix[y_gold]
		pred_suffix = label2suffix[y_guess]
		word_type = "".join(["c" if seg in CONS else "v" for seg in word.split(" ")])
		assert word_type in word_types
		is_correct = 1 if gold_suffix == pred_suffix else 0
		acc_dict[word_type][gold_suffix].append(is_correct)
		if not is_correct:
			error_dict[word_type][gold_suffix].append((word, y_gold, y_guess))

	# Transforms list of 1s and 0s in acc dict into accuracy scalar
	for word_type in acc_dict:
		for gold_label in acc_dict[word_type]:
			accs_list = acc_dict[word_type][gold_label]
			if len(accs_list):
				accs = sum(accs_list)/len(accs_list)
			else:
				accs = -1
			acc = round(accs, 3)
			acc_dict[word_type][gold_label] = f"Acc: {acc} Num_correct: {sum(accs_list)} Num_total: {len(accs_list)}" # type: ignore

	# Format and write results to file
	lines = []
	lines.append("\n######## Accuracies ########\n")
	for word_type in acc_dict:
		lines.append(f"----------- {word_type} -----------\n")
		for gold_label, acc in acc_dict[word_type].items():
			lines.append(f"{gold_label}: {acc}\n")

	lines.append("\n######## Predictions ########\n")
	lines.append("singular - gold suffix - predicted suffix\n")
	for word_type in error_dict:
		lines.append(f"----------- {word_type} -----------\n")
		for gold_label in error_dict[word_type]:
			lines.append(f"{gold_label}\n")
			for word, y_gold, y_guess in error_dict[word_type][gold_label]:
				gold_suffix = label2suffix[y_gold]
				pred_suffix = label2suffix[y_guess]
				lines.append(f"   {word} - {gold_suffix} - {pred_suffix}\n")

	with open(write_filepath, "w", encoding="utf-8") as f:
		f.writelines(lines)


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

	TRAINING_DATA_FOLDER = "EqualDefault"
	FILE_PREFIX = "equalFreq"
	POOLING_FUNC = pool_last # TODO change pooling function
	POOLING_FUNC_name = "pool_last"
	NUM_ITERS = 300
	BATCH_SIZE = 3

	TRAINING_DATA = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_train.txt"

	# Train classifier
	train_SGs, train_PLs, train_Ls = process_file(TRAINING_DATA)
	X_train, y_train = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC)
	classes = np.unique(y_train)
	class_1_accs, class_2_accs, class_3_accs = [], [], []

	model = SGDClassifier(loss="log_loss", max_iter=1, tol=None, warm_start=True, eta0=0.05, learning_rate="constant") # TODO adjust parameters of model

	num_batches = int(np.ceil(NUM_ITERS/BATCH_SIZE))

	for i in range(0, NUM_ITERS, BATCH_SIZE):
		X_batch = X_train[i: i+BATCH_SIZE]
		y_batch = y_train[i: i+BATCH_SIZE]
		model.partial_fit(X_batch, y_batch, classes=classes)

		test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_test.txt" # TODO change for learning curves of different conditions

		test_SGs, test_PLs, test_Ls = process_file(test_filepath)
		X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, override_max_syll=5)
		y_pred = model.predict(X_test)
		acc_dict = calc_results_by_gold_label(y_test, y_pred)

		class_1_accs.append(acc_dict["W AH0"])
		class_2_accs.append(acc_dict["L EY0"])
		class_3_accs.append(acc_dict["Y IY0"])

	# Plot learning curve
	iterations = [i for i in range(num_batches)]
	plot_learning_curve(class_1_accs, class_2_accs, class_3_accs, iterations)


	# Plot weight heatmap
	heatmap_save_path = f"{TRAINING_DATA_FOLDER}_LR_{POOLING_FUNC_name}.jpg" 
	get_heatmaps(model, POOLING_FUNC_name, heatmap_save_path)
	

	test_file_suffixes = ["test", "test_H", "test_L", "test_Mutants", "testNewTemplates"]
	# test_file_suffixes = ["test"]
	for test_file_suffix in test_file_suffixes:
		print(f"Testing {test_file_suffix}")
		test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_{test_file_suffix}.txt"

		write_filepath = f"./{TRAINING_DATA_FOLDER}_Results/{FILE_PREFIX}_{test_file_suffix}_{POOLING_FUNC_name}_RESULT.txt" 

		test_SGs, test_PLs, test_Ls = process_file(test_filepath)
		X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, override_max_syll=5)
		y_pred = model.predict(X_test)
		acc_dict = calc_results_by_gold_label(y_test, y_pred)

		print(f"Current test file suffix: {test_file_suffix}")
		print(f'{acc_dict["W AH0"]} - {acc_dict["L EY0"]} - {acc_dict["Y IY0"]}')

		write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath)
