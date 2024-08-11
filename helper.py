import numpy as np
from re import sub
from random import shuffle

# ------------ Preprocessing functions ------------ # 
"""Initialize feature and category dictionaries."""

def init_resource_dicts(feature_filepath):
	feat_file = open(feature_filepath, "r")
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
	return symbol2feats, suffix2label, label2suffix


"""Functions for reading files and processing data. Adapated from Brandon's LSTM code."""

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


def shuffle_arrays(sgs, pls, lengths):
	combined = list(zip(sgs, pls, lengths))
	shuffle(combined) 
	list1_shuffled, list2_shuffled, list3_shuffled = zip(*combined) 
	return list(list1_shuffled), list(list2_shuffled), list(list3_shuffled)


def process_file(filepath):
	"""Reads a data file into three lists, singular forms, plural forms, and lengths. Also shuffles them"""
	sgs, pls, lengths = get_strings(filepath)
	sgs, pls, lengths = shuffle_arrays(sgs, pls, lengths)
	return sgs, pls, lengths


def get_arrays(UR_strings, SR_strings, syll_lengths, symbol2feats, suffix2label, pool_func, override_max_syll=0):
	"""
	symbol2feats: dict[segment -> vector of features]
	suffix2label: dict[suffix -> int from {0, 1, 2}] 
	
	Transformed processed data from files into vectors. Pools X using pool_func.
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
	X = pool_func(X)

	Y = np.array(Y_list) # single int
	return X, Y