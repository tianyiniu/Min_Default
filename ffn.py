"""Use a traditional neural network setup to classify plural class."""

from re import sub
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from matplotlib import pyplot as plt

from random import shuffle


CONS = ["P", "B", "T", "D", "K", "G", "NG", "M", "N", "L", "F",  "V", "S", "Z", "SH", "ZH", "CH", "JH", "H"]
VOWELS = ["IH0", "EH0", "AH0", "UH0", "IY0", "UW0", "EY0", "OW0", "IH1", "EH1", "AH1", "UH1", "IY1", "UW1", "EY1", "OW1", "IH2", "EH2", "AH2", "UH2", "IY2", "UW2", "EY2", "OW2"]
FEATURES_FILE = "featsNew"

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


# Define the FFN model
class SimpleFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
	

def train_network(train_SGs, train_PLs, train_Ls, num_epochs=10):


URs, SRs, Ls = get_strings(TRAINING_DATA)
X, y = get_arrays(URs, SRs, Ls, symbol2feats, suffix2label)

print(X.shape)
X = POOLING_FUNC(X)
print(X.shape)
# Split the dataset into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = one_hot(torch.tensor(y_train).to(torch.int64), num_classes=3).to(torch.float32)


# Prepare data loaders
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[-1] # TODO make variable

hidden_dim = 100
output_dim = 3

model = SimpleFFN(input_dim, hidden_dim, output_dim)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10

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
	
# Define the inference function
def inference(model, inputs):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to compute gradients for inference
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        predictions_index = torch.argmax(predictions, dim=1)
    return predictions_index

X_valid = torch.tensor(X_valid, dtype=torch.float32)

# Perform inference
y_pred = inference(model, X_valid)
class_report = classification_report(y_valid, y_pred)
print(class_report)

	

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
	# get_heatmaps(model, POOLING_FUNC_name, heatmap_save_path)
	

	# test_file_suffixes = ["test", "test_H", "test_L", "test_Mutants", "testNewTemplates"]
	test_file_suffixes = ["test"]
	for test_file_suffix in test_file_suffixes:
		print(f"Testing {test_file_suffix}")
		test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_{test_file_suffix}.txt"

		if WITH_ISLANDS:
			write_filepath = f"./{TRAINING_DATA_FOLDER}_islands_Results/{FILE_PREFIX}_{test_file_suffix}_{POOLING_FUNC_name}_RESULTS.txt"
		else:
			write_filepath = f"./{TRAINING_DATA_FOLDER}_Results/{FILE_PREFIX}_{test_file_suffix}_{POOLING_FUNC_name}_RESULTS.txt"

		test_SGs, test_PLs, test_Ls = process_file(test_filepath)
		y_test, y_pred = test_classifier(test_SGs, test_PLs, test_Ls)
		acc_dict = calc_results_by_gold_label(y_test, y_pred)

		class_1_accs.append(acc_dict["W AH0"])
		class_2_accs.append(acc_dict["L EY0"])
		class_3_accs.append(acc_dict["Y IY0"])

		# write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath)

	print(class_1_accs)
	print(class_2_accs)
	print(class_3_accs)
