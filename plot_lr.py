import sys
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from helper import * 
from pooling_functions import *

from tqdm import tqdm

def plot_learning_curve_full(accs, iterations, save_filepath):
	# plt.figure(figsize=(10, 6))
	plt.figure(figsize=(10, 6))
	plt.style.use('ggplot')
	
	# Define line styles and colors for each condition
	line_styles = {
		"Suffix A": 'dotted',
		"Suffix B": 'dashed',
		"Suffix C": 'solid'
	}

	colors = {
		"Minority Default": "red",
		"Equal Frequency": "blue",
		"Majority Default": "orange"
	}
	
	suffixes = ['Suffix A', 'Suffix B', 'Suffix C']
	default_conditions = ["Minority Default", "Equal Frequency", "Majority Default"]
	
	for suffix in suffixes:
		for condition in default_conditions:
			style = line_styles[suffix]
			color = colors[condition]
			plt.plot(iterations, accs[condition][suffix], linestyle=style, color=color)

	plt.title(f'Learning curve of logistic regression (pool-last)', fontsize=20, pad=20) # TODO Change pool func
	plt.xlabel('Number of batches', fontsize=16)
	plt.ylabel('Average Proportion Correct', fontsize=16)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylim(0, 1.01)
	plt.xlim(0, max(iterations))

	
	# Create custom legends for both suffixes and conditions
	suffix_legend_lines = [plt.Line2D([0], [0], color='black', linestyle=line_styles[suffix], lw=2) for suffix in suffixes]
	suffix_legend_labels = suffixes
	
	condition_legend_lines = [plt.Line2D([0], [0], color=colors[condition], lw=2) for condition in default_conditions]
	condition_legend_labels = default_conditions
	
	legend1 = plt.legend(suffix_legend_lines, suffix_legend_labels, bbox_to_anchor=(1, 0.4), loc='lower left', title="Correct Suffix", frameon=False, fontsize=12)
	legend1.set_title("Correct Suffix", prop={'size':16})
	legend1._legend_box.align = "left"
	legend2 = plt.legend(condition_legend_lines, condition_legend_labels, bbox_to_anchor=(1, 0.1), loc='lower left', title="Condition", frameon=False, fontsize=12)
	legend2.set_title("Condition", prop={'size': 16})
	legend2._legend_box.align = "left"
	
	plt.gca().add_artist(legend1)
	plt.gca().add_artist(legend2)

	plt.subplots_adjust(right=0.8)
	plt.savefig(save_filepath, format="jpg", dpi=300)


if __name__ == "__main__":	

	# ------------ Initialize file paths, resource dictionaries ------------ # 

	CONS = ["P", "B", "T", "D", "K", "G", "NG", "M", "N", "L", "F",  "V", "S", "Z", "SH", "ZH", "CH", "JH", "H"]
	VOWELS = ["IH0", "EH0", "AH0", "UH0", "IY0", "UW0", "EY0", "OW0", "IH1", "EH1", "AH1", "UH1", "IY1", "UW1", "EY1", "OW1", "IH2", "EH2", "AH2", "UH2", "IY2", "UW2", "EY2", "OW2"]

	FEATURES_FILE = "Feature_files/featsNew"
	FEATURE_NAMES = ["cons", "syll", "son", "approx", "voice", "cont", "nas", "strid", "lab", "cor", "ant", "dist", "dor", "high", "back", "tense", "diph", "stress", "main"]


	symbol2feats, suffix2label, label2suffix = init_resource_dicts(FEATURES_FILE)

	# ------------ Model hyperparameters ------------ # 
	TS = [("EqualDefault", "equalFreq"), ("MajDefault", "majDefault"), ("MinDefault", "minDefault")]
	MODEL_NAME = "LR"

	POOLING_FUNC = pool_last # TODO change pooling function
	POOLING_FUNC_name = "pool_last"

	# POOLING_FUNC = pool_concat # TODO change pooling function
	# POOLING_FUNC_name = "pool_concat"
	
	acc_curves_dict = {}
	for TRAINING_DATA_FOLDER, FILE_PREFIX in TS:
		
		WRITE_RESULT_FOLDER = f"./Pool_last_results/{TRAINING_DATA_FOLDER}_{POOLING_FUNC_name}_Results_{MODEL_NAME}"
		check_dir_exists(WRITE_RESULT_FOLDER) 

		BATCH_SIZE = 10
		# TODO change back to 3
		NUM_EPOCHS = 3
		# TODO change back to 10 
		NUM_REPEATS = 10 # Repeat LR model training 10 times for more reliable accuracy curve
		LEARNING_RATE = 0.01 # 0.01 for pool last, 0.05 for pool concat

		VALIDATION_FILE_PREFIX = "test"

		train_data_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_train.txt"

		class_1_accs, class_2_accs, class_3_accs = [], [], []
		for i in tqdm(range(NUM_REPEATS)):

			# Train classifier
			train_SGs, train_PLs, train_Ls = process_file(train_data_filepath)
			X_train_org, y_train_org = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC)
			# Copy and concatenation X_train and y_train to simulate epochs. Doing so prevents deeply nested training loops
			X_train = np.concatenate([X_train_org] * NUM_EPOCHS, axis=0)
			y_train = np.concatenate([y_train_org] * NUM_EPOCHS, axis=0)
			classes = np.unique(y_train)

			run_class_1_accs, run_class_2_accs, run_class_3_accs = [], [], []

			model = SGDClassifier(loss="log_loss", max_iter=1, tol=None, warm_start=True, eta0=LEARNING_RATE, learning_rate="constant") # TODO adjust parameters of model

			curr_batch_num = 0
			for j in range(0, X_train.shape[0], BATCH_SIZE):

				X_batch = X_train[j: j+BATCH_SIZE]
				y_batch = y_train[j: j+BATCH_SIZE]
				model.partial_fit(X_batch, y_batch, classes=classes)

				test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_test.txt"

				test_SGs, test_PLs, test_Ls = process_file(test_filepath)
				X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, override_max_syll=5)
				y_pred = model.predict(X_test)
				acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)

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

		acc_dict_names = {"equalFreq": "Equal Frequency", "majDefault": "Majority Default", "minDefault": "Minority Default"}

		acc_curves_dict[acc_dict_names[FILE_PREFIX]] = {
								"Suffix A": avg_class_1_accs, 
								"Suffix B": avg_class_2_accs, 
								"Suffix C": avg_class_3_accs
								}


	# Plot learning curve
	curve_save_path = f"pool_last_CURVE_ALL.jpg" # TODO Change pool func 
	num_batches = int(np.ceil(X_train.shape[0]/BATCH_SIZE))
	iterations = [i for i in range(len(avg_class_1_accs))]
	plot_learning_curve_full(acc_curves_dict, iterations, curve_save_path) 
