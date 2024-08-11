import sys
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from helper import * 
from pooling_functions import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------ Define FFN class and relevant functions ------------ # 

class SimpleFFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50):
        super(SimpleFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

def inference(model, inputs):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to compute gradients for inference
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        predictions_index = torch.argmax(predictions, dim=1)
    return predictions_index


# ------------ Functions for making plots ------------ #

def plot_learning_curve(class_1_accs, class_2_accs, class_3_accs, iterations, save_filepath="learning_curve_temp.jpg"):
	plt.figure(figsize=(10, 6))
	
	# Plotting each accuracy list
	plt.plot(iterations, class_1_accs, label='W AH0', color="red")
	plt.plot(iterations, class_2_accs, label='L EY0', color="yellow")
	plt.plot(iterations, class_3_accs, label='Y IY0', color="blue")
	
	# Adding titles and labels
	plt.title(f'Learning Curve - FFN - Equal Default - EqualFreq - {POOLING_FUNC_name}')
	plt.xlabel('Num epochs')
	plt.ylabel('Accuracy')
	
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

	TRAINING_DATA_FOLDER = "EqualDefault"
	FILE_PREFIX = "equalFreq"
	POOLING_FUNC = pool_last # TODO change pooling function
	POOLING_FUNC_name = "pool_last"
	MODEL_NAME = "FFN"

	train_data_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_train.txt"

	# Train classifier
	train_SGs, train_PLs, train_Ls = process_file(train_data_filepath)
	X_train, y_train = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC)
	classes = np.unique(y_train)
	class_1_accs, class_2_accs, class_3_accs = [], [], []

	# Convert to PyTorch tensors
	X_train = torch.tensor(X_train, dtype=torch.float32)
	y_train = one_hot(torch.tensor(y_train).to(torch.int64), num_classes=3).to(torch.float32)

	# Prepare data loaders
	dataset = TensorDataset(X_train, y_train)
	train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

	# Initialize the model, loss function, and optimizer
	input_dim = X_train.shape[-1] 
	output_dim = y_train.shape[-1]
	print(f"Input dim: {input_dim} Output dim: {output_dim}")

	model = SimpleFFN(input_dim, output_dim, hidden_dim=100)
	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	# Train the model
	num_epochs = 20
	class_1_accs, class_2_accs, class_3_accs = [], [], []
	for epoch in range(num_epochs):
		for inputs, labels in train_loader:
			# Forward pass
			outputs = model(inputs)
			loss = loss_fn(outputs, labels)

			# Backward pass and optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

		test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_test.txt" # TODO change for learning curves of different conditions

		# Process file, convert to vector, then convert X to tensor, Y to one-hot tensor
		test_SGs, test_PLs, test_Ls = process_file(test_filepath)
		X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, override_max_syll=5)
		X_test = torch.tensor(X_test, dtype=torch.float32)
		y_test_one_hot = one_hot(torch.tensor(y_test).to(torch.int64), num_classes=3).to(torch.float32)

		# Predict and calculate accs by gold label
		y_pred = inference(model, X_test).tolist()
		acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)

		class_1_accs.append(acc_dict["W AH0"])
		class_2_accs.append(acc_dict["L EY0"])
		class_3_accs.append(acc_dict["Y IY0"])


	print(f'{acc_dict["W AH0"]} - {acc_dict["L EY0"]} - {acc_dict["Y IY0"]}')
	epochs = [i for i in range(num_epochs)]
	plot_learning_curve(class_1_accs, class_2_accs, class_3_accs, epochs, save_filepath="temp_ffn_learning_curve.jpg")

	test_file_suffixes = ["test", "test_H", "test_L", "test_Mutants", "testNewTemplates"]
	# test_file_suffixes = ["test"]
	for test_file_suffix in test_file_suffixes:
		print(f"Testing {test_file_suffix}")
		test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_{test_file_suffix}.txt"
		write_filepath = f"./{TRAINING_DATA_FOLDER}_Results_{MODEL_NAME}/{FILE_PREFIX}_{test_file_suffix}_{POOLING_FUNC_name}_RESULT.txt" 

		test_SGs, test_PLs, test_Ls = process_file(test_filepath)
		X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, override_max_syll=5)
		X_test = torch.tensor(X_test, dtype=torch.float32)
		y_test_one_hot = one_hot(torch.tensor(y_test).to(torch.int64), num_classes=3).to(torch.float32)

		# Predict and calculate accs by gold label
		y_pred = inference(model, X_test).tolist()
		acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)

		print(f'{acc_dict["W AH0"]} - {acc_dict["L EY0"]} - {acc_dict["Y IY0"]}')

		write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath, suffix2label, label2suffix)
