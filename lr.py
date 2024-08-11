import sys
from re import sub
import numpy as np
import pandas as pd 
import seaborn as sns
from random import shuffle
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

CONS = ["P", "B", "T", "D", "K", "G", "NG", "M", "N", "L", "F",  "V", "S", "Z", "SH", "ZH", "CH", "JH", "H"]
VOWELS = ["IH0", "EH0", "AH0", "UH0", "IY0", "UW0", "EY0", "OW0", "IH1", "EH1", "AH1", "UH1", "IY1", "UW1", "EY1", "OW1", "IH2", "EH2", "AH2", "UH2", "IY2", "UW2", "EY2", "OW2"]

FEATURES_FILE = "featsNew"
FEATURE_NAMES = ["cons", "syll", "son", "approx", "voice", "cont", "nas", "strid", "lab", "cor", "ant", "dist", "dor", "high", "back", "tense", "diph", "stress", "main"]

# FEATURES_FILE = "featsNew_mini"
# FEATURE_NAMES = ["son", "voice", "strid"]

# ------------ Preprocessing Functions ------------ # 

"""Read features.txt and initialize feature dictionaries. Adapated from Brandon's LSTM code."""

def get_strings(data_file):
	"""
	Process input file into a list of strings.
	Returns: 
		UR: ["UW0 B IH1 CH", "L AH0 D AH1 F", ...]
		SR: ["W AH0", "L EY0", ...]
		syll_lengs: [4, 5, ...]
	"""
	with open(data_file, "r", encoding="utf-8") as f:
		f.readline() # Skip first line i.e., "singular,plural\n"
		UR_strings, SR_strings, syll_lengths = [], [], []
		ur_num = 0

		for line in f.readlines():
			columns = line.rstrip().split(",")
			if len(columns) == 2:
				ur, sr = columns
				if sr == "" or ur == "":
					continue
				ur_num += 1

				syll_lengths.append(len([seg for seg in ur.split(" ") if seg != ""])) # Number of segmentsin singular

				UR_strings.append(ur)
				SR_strings.append(sr[-5:]) # Last 5 characters correspond to plural suffix

			else:
				print(line)
				raise Exception("Training data error! All lines should have 2 columns in TD files!")
			
	return UR_strings, SR_strings, syll_lengths


def get_arrays(UR_strings, SR_strings, syll_lengths, symbol2feats, suffix2label, override_max_syll=0):
	"""
	Process input file into a list of strings.
	symbol2feats: dict[segment -> vector of features]
	suffix2label: dict[suffix -> int from {0, 1, 2}] 
	"""

	if override_max_syll:
		assert override_max_syll >= max(syll_lengths)
		max_len = override_max_syll
	else: 
		max_len = max(syll_lengths)

	X_list, Y_list = [], []
 
	for word_index, syll_length in enumerate(syll_lengths):
		padding = " ".join(["_"]*(max_len-syll_length))
		this_ur = UR_strings[word_index]+" "+padding # Singular form + padding as string
		this_sr = SR_strings[word_index][-5:] # Suffix as string

		# TODO Fix some errors in data files (still necessary?):
		this_ur = sub(" J ", " Y ", this_ur)
		this_ur = sub(" C ", " CH ", this_ur)

		X_list.append([symbol2feats[seg] for seg in this_ur.split(" ") if seg != ""])
		Y_list.append(suffix2label[this_sr])

	X = np.array(X_list) # list of vectors
	Y = np.array(Y_list) # single int
	return X, Y

def shuffle_arrays(sgs, pls, lengths):
	combined = list(zip(sgs, pls, lengths))
	shuffle(combined) 
	list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined) 
	return list(list1_shuffled), list(list2_shuffled), list(list3_shuffled)
	

# ------------ Pooling Functions ------------ # 
def pool_average(X):
	"""Pools phonetic feature vectors by averaging across all segments."""
	# X.shape (n, 5, 19)
	return np.mean(X, axis=1)

def pool_sum(X):
	"""Pools phonetic feature vectors by summation across all segments."""
	return np.sum(X, axis=1)

def pool_concat(X):
	"""Pools features by concat each features vector head-to-tail. Results in a word-level feature vector of 5x19"""
	return np.array([np.concatenate(submatrices, axis=0) for submatrices in X])

def pool_last(X): 
	"""Pools phonetic feature vectors by only returning the final segment. Code must correctly identify last non-padding segment"""
	new_X = []
	max_segments = len(X[0])
	for word in X:
		curr_idx = max_segments - 1
		while curr_idx >= 0:
			last = word[curr_idx]
			if len(set(last)) > 1: # Not pad token
				break
			curr_idx -= 1
		new_X.append(last)
	return np.array(new_X)

feat_file = open(FEATURES_FILE, "r")
feat_names = feat_file.readline().rstrip().split("\t")[1:] # First line specifies feature names
symbol2feats = {'_': [0.0 for f in feat_names]}

for line in feat_file.readlines():
	columns = line.rstrip().split("\t") 
	seg = columns[0]
	values = [{"-":-1.0, "+":1.0, "0":0.0}[v] for v in columns[1:]]
	symbol2feats[seg] = values

suffix2label = {
	"W AH0": 0, #wuh
	"L EY0": 1, #lay
	"Y IY0": 2 #yee
}

label2suffix = {
	0: "W AH0", #wuh
	1: "L EY0", #lay
	2: "Y IY0" #yee
}

# ------------ Train Classifier ------------ #
def process_file(filepath):
	sgs, pls, lengths = get_strings(filepath)
	sgs, pls, lengths = shuffle_arrays(sgs, pls, lengths)
	return sgs, pls, lengths

def train_classifier(train_SGs, train_PLs, train_Ls, num_iters=100):
	X_train, y_train = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label)
	X_train = POOLING_FUNC(X_train) 

	# Initialize and train the logistic regression model
	model = LogisticRegression(max_iter=num_iters)
	model.fit(X_train, y_train)

	return model


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

# ------------ Test Classifier ------------ #
def test_classifier(test_SGs, test_PLs, test_Ls):
	X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, override_max_syll=5)

	X_test = POOLING_FUNC(X_test)

	y_pred = model.predict(X_test)
	acc_score = accuracy_score(y_test, y_pred)
	return y_test, y_pred

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
	word_types = set()
	for sg in test_SGs:
		segments = sg.split(" ")
		word_type = ["c" if seg in CONS else "v" for seg in segments]
		word_types.add("".join(word_type))

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
			
	for word_type in acc_dict:
		for gold_label in acc_dict[word_type]:
			accs_list = acc_dict[word_type][gold_label]
			if len(accs_list):
				accs = sum(accs_list)/len(accs_list)
			else:
				accs = -1
			acc = round(accs, 3)
			acc_dict[word_type][gold_label] = f"Acc: {acc} Num_correct: {sum(accs_list)} Num_total: {len(accs_list)}" # type: ignore

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
	TRAINING_DATA_FOLDER = "EqualDefault"
	FILE_PREFIX = "equalFreq"
	WITH_ISLANDS = False
	POOLING_FUNC = pool_last # TODO change pooling function
	POOLING_FUNC_name = "pool_last"
	NUM_ITERS = 300

	if WITH_ISLANDS:
		TRAINING_DATA = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_train_withIslands.txt"
	else:
		TRAINING_DATA = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_train.txt"

	# Train classifier
	train_SGs, train_PLs, train_Ls = process_file(TRAINING_DATA)

	class_1_accs = []
	class_2_accs = []
	class_3_accs = []

	model = train_classifier(train_SGs, train_PLs, train_Ls, num_iters=NUM_ITERS)

	# Extract and save model weights as heatmap
	if WITH_ISLANDS:
		heatmap_save_path = f"{TRAINING_DATA_FOLDER}_islands_LR_{POOLING_FUNC_name}.jpg"
	else:
		heatmap_save_path = f"{TRAINING_DATA_FOLDER}_LR_{POOLING_FUNC_name}.jpg" 
	get_heatmaps(model, POOLING_FUNC_name, heatmap_save_path)
	

	test_file_suffixes = ["test", "test_H", "test_L", "test_Mutants", "testNewTemplates"]
	# test_file_suffixes = ["test"]
	for test_file_suffix in test_file_suffixes:
		print(f"Testing {test_file_suffix}")
		test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_{test_file_suffix}.txt"

		if WITH_ISLANDS:
			write_filepath = f"./{TRAINING_DATA_FOLDER}_islands_Results/{FILE_PREFIX}_{test_file_suffix}_{POOLING_FUNC_name}_RESULTS.txt"
		else:
			write_filepath = f"./{TRAINING_DATA_FOLDER}_Results/{FILE_PREFIX}_{test_file_suffix}_{POOLING_FUNC_name}_RESULT.txt" 

		test_SGs, test_PLs, test_Ls = process_file(test_filepath)
		y_test, y_pred = test_classifier(test_SGs, test_PLs, test_Ls)
		acc_dict = calc_results_by_gold_label(y_test, y_pred)

		class_1_accs.append(acc_dict["W AH0"])
		class_2_accs.append(acc_dict["L EY0"])
		class_3_accs.append(acc_dict["Y IY0"])

		print(f"Current test file suffix: {test_file_suffix}")
		print(f'{acc_dict["W AH0"]} - {acc_dict["L EY0"]} - {acc_dict["Y IY0"]}')

		write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath)
