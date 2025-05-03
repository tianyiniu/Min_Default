"""Defines a series of helper functions and mappings"""

import os
import numpy as np
import matplotlib.pyplot as plt

from re import sub
from random import shuffle

CONS = ["P", "B", "T", "D", "K", "G", "NG", "M", "N", "L", "F",  "V", "S", "Z", "SH", "ZH", "CH", "JH", "H"]
VOWELS = ["IH0", "EH0", "AH0", "UH0", "IY0", "UW0", "EY0", "OW0", "IH1", "EH1", "AH1", "UH1", "IY1", "UW1", "EY1", "OW1", "IH2", "EH2", "AH2", "UH2", "IY2", "UW2", "EY2", "OW2"]

def check_dir_exists(directory_path):
    """Check if the directory exists at directory_path. If not, create the directory"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# ------------ Preprocessing functions ------------ # 

def init_resource_dicts(feature_filepath):
    """Initialize feature and category dictionarie:
        symbol2feats maps phonemes to feature vector
        suffix2label maps string suffix to numerical category
        label2suffix inverses suffix2label
    """
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


def get_arrays(UR_strings, SR_strings, syll_lengths, symbol2feats, suffix2label, pool_func=None, override_max_syll=0, padding_loc="back"):
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
        if padding_loc == "back":
            this_ur = UR_strings[word_index]+" "+padding # Singular form + padding as string
        else: 
            this_ur = padding+" "+UR_strings[word_index]

        this_sr = SR_strings[word_index][-5:] # Suffix as string

        # TODO Fix some errors in data files (still necessary?):
        this_ur = sub(" J ", " Y ", this_ur)
        this_ur = sub(" C ", " CH ", this_ur)

        X_list.append([symbol2feats[seg] for seg in this_ur.split(" ") if seg != ""])
        Y_list.append(suffix2label[this_sr])

    X = np.array(X_list) # list of vectors
    if pool_func:
        X = pool_func(X)
 
    Y = np.array(Y_list) # single int
    return X, Y


# ------------ Test Accuracy by Gold Label ------------ #
def calc_results_by_gold_label(y_test, y_pred, suffix2label=None, label2suffix=None):
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

# ------------ Plotting functions ------------ # 

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
            print(len(iterations))
            print(len(accs[condition][suffix]))
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


def plot_learning_curve_full2(accs, iterations, save_filepath):
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
    
    # Create a cleaner naming for the legend
    legend_names = {
        "Minority Default": "Minority Default", 
        "Equal Frequency": "Equal Frequency", 
        "Majority Default": "Majority Default"
    }
    
    suffixes = ['Suffix A', 'Suffix B', 'Suffix C']
    default_conditions = ["Minority Default", "Equal Frequency", "Majority Default"]
    
    for suffix in suffixes:
        for condition in default_conditions:
            style = line_styles[suffix]
            color = colors[condition]
            plt.plot(iterations, accs[condition][suffix], linestyle=style, color=color)
    
    plt.title('Learning curve of logistic regression (pool-last)', fontsize=20, pad=20)
    plt.xlabel('Number of batches', fontsize=16)
    plt.ylabel('Average Proportion Correct', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1.01)
    plt.xlim(0, max(iterations))
    
    # Create custom legends with better positioning
    suffix_legend_lines = [plt.Line2D([0], [0], color='black', linestyle=line_styles[suffix], lw=2) for suffix in suffixes]
    suffix_legend_labels = suffixes
    
    condition_legend_lines = [plt.Line2D([0], [0], color=colors[condition], lw=2) for condition in default_conditions]
    condition_legend_labels = [legend_names[condition] for condition in default_conditions]
    
    # Position the legend outside the plot
    # Create a figure with a specific layout to accommodate the legend
    plt.subplots_adjust(right=0.75)  # This leaves 25% of space on the right for legends
    
    # Add legends with proper positioning
    legend1 = plt.legend(suffix_legend_lines, suffix_legend_labels, 
                         bbox_to_anchor=(1.05, 0.7), loc='upper left', 
                         title="Correct Suffix", frameon=False, fontsize=12)
    legend1.set_title("Correct Suffix", prop={'size': 14})
    
    plt.gca().add_artist(legend1)
    
    legend2 = plt.legend(condition_legend_lines, condition_legend_labels, 
                         bbox_to_anchor=(1.05, 0.3), loc='upper left', 
                         title="Condition", frameon=False, fontsize=12)
    legend2.set_title("Condition", prop={'size': 14})
    
    # Save the figure with tight bounding box to ensure everything is included
    plt.savefig(save_filepath, format="png", dpi=300, bbox_inches="tight")


# ------------ Test Accuracy by Word Type ------------ #
def write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath, suffix2label=None, label2suffix=None):
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