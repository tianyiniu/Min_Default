import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from helper import * 
from tqdm import tqdm
from pooling_functions import *
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


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

        class_names = ["Suffix A", "Suffix B", "Suffix C"]
        weights_df = pd.DataFrame(weights, columns=feature_names,
                                index=class_names).round(2)
        
        # TODO remove later Keep only positions one and two 
        # weights_df = weights_df.loc[:, weights_df.columns.str.endswith('0') | weights_df.columns.str.endswith('1')]

        fig, ax = plt.subplots(figsize=(30, 8))
        sns.heatmap(weights_df, annot=True, cmap='coolwarm', center=0, cbar=False, linewidths=1, linecolor='black', square=True, cbar_kws={"shrink":0.3})
    else: 
        feature_names = FEATURE_NAMES
        class_names = ["Suffix A", "Suffix B", "Suffix C"]
        weights_df = pd.DataFrame(weights, columns=feature_names,
                                index=class_names).round(2)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(weights_df, annot=True, cmap='coolwarm', center=0, cbar=False, linewidths=1, linecolor='black', square=True)

    default_names = {"MajDefault": "Majority Default", "EqualDefault": "Equal Frequency", "MinDefault": "Minority Default"}

    plt.title(f'Logistic Regression Weights - {default_names[TRAINING_DATA_FOLDER]} ({POOLING_FUNC_name})')
    plt.xticks(rotation=45)
    plt.savefig(save_filepath, format="jpg", dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved heatmap to path: {save_filepath}")


if __name__ == "__main__":	

    #------Initialize file paths, resource dictionaries------# 

    CONS = ["P", "B", "T", "D", "K", "G", "NG", "M", "N", "L", "F",  "V", "S", "Z", "SH", "ZH", "CH", "JH", "H"]
    VOWELS = ["IH0", "EH0", "AH0", "UH0", "IY0", "UW0", "EY0", "OW0", "IH1", "EH1", "AH1", "UH1", "IY1", "UW1", "EY1", "OW1", "IH2", "EH2", "AH2", "UH2", "IY2", "UW2", "EY2", "OW2"]

    FEATURES_FILE = "Feature_files/featsNew"
    FEATURE_NAMES = ["cons", "syll", "son", "approx", "voice", "cont", "nas", "strid", "lab", "cor", "ant", "dist", "dor", "high", "back", "tense", "diph", "stress", "main"]

    symbol2feats, suffix2label, label2suffix = init_resource_dicts(FEATURES_FILE)

    # ------------ Model hyperparameters ------------ # 
    TS = [("EqualDefault", "equalFreq"), ("MajDefault", "majDefault"), ("MinDefault", "minDefault")]
    MODEL_NAME = "LR"

    POOLING_FUNC = pool_last
    POOLING_FUNC_name = "pool_last"

    acc_curves_dict = {}
    for TRAINING_DATA_FOLDER, FILE_PREFIX in TS:
    
        WRITE_RESULT_FOLDER = f"./Data/LR_PL/{TRAINING_DATA_FOLDER}_Results"
        check_dir_exists(WRITE_RESULT_FOLDER) 

        BATCH_SIZE = 10
        NUM_EPOCHS = 3
        NUM_REPEATS = 10 
        LEARNING_RATE = 0.01

        train_data_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_train.txt"
        test_filepath = f"./{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_test.txt"


        class_1_accs, class_2_accs, class_3_accs = [], [], []
        for i in tqdm(range(NUM_REPEATS)):

            # Train classifier
            train_SGs, train_PLs, train_Ls = process_file(train_data_filepath)

            X_train_org, y_train_org = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, padding_loc="front")

            X_train = np.concatenate([X_train_org] * NUM_EPOCHS, axis=0)
            y_train = np.concatenate([y_train_org] * NUM_EPOCHS, axis=0)
            classes = np.unique(y_train)

            run_class_1_accs, run_class_2_accs, run_class_3_accs = [], [], []

            model = SGDClassifier(loss="log_loss", max_iter=1, tol=None, warm_start=True, eta0=LEARNING_RATE, learning_rate="constant")

            curr_batch_num = 0
            for j in range(0, X_train.shape[0], BATCH_SIZE):

                X_batch = X_train[j: j+BATCH_SIZE]
                y_batch = y_train[j: j+BATCH_SIZE]
                model.partial_fit(X_batch, y_batch, classes=classes)


                test_SGs, test_PLs, test_Ls = process_file(test_filepath)
                X_test, y_test = get_arrays(test_SGs, test_PLs, test_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC, override_max_syll=5, padding_loc="front")
                y_pred = model.predict(X_test)
                acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)


                #     write_filepath = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_RESULT_{curr_batch_num}.txt" 
                #     acc_dict = calc_results_by_gold_label(y_test, y_pred, suffix2label, label2suffix)
                #     write_results_by_word_type(test_SGs, y_test, y_pred, write_filepath, suffix2label, label2suffix)

                run_class_1_accs.append(acc_dict["W AH0"])
                run_class_2_accs.append(acc_dict["L EY0"])
                run_class_3_accs.append(acc_dict["Y IY0"])

                curr_batch_num += 1

            # Get model's accuracy on test markers
            for test_condition in ["train", "test", "test_Mutants", "testNewTemplates", "test_H", "test_L"]:
                fp = f"{TRAINING_DATA_FOLDER}/{FILE_PREFIX}_{test_condition}.txt"
                sgs, pls, ls = process_file(fp)
                x, y = get_arrays(sgs, pls, ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC)
                y_hat = model.predict(x)
                acc = accuracy_score(y, y_hat)
                print(f"{FILE_PREFIX}, {test_condition}: {acc}")


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

        heatmap_save_path = f"LR-PL_{FILE_PREFIX}_heatmap_test.jpg"
        get_heatmaps(model, POOLING_FUNC_name, heatmap_save_path)

    # Plot learning curve
    curve_save_path = f"LR-PL_curve_all_front.png"
    num_batches = int(np.ceil(X_train.shape[0]/BATCH_SIZE))
    iterations = [i for i in range(len(avg_class_1_accs)+1)] 

    plot_acc_curves_dict = {}
    for mintype, mintype_dict in acc_curves_dict.items():
        plot_acc_curves_dict[mintype] = {}
        for suffix, suffixlst in mintype_dict.items():
            plot_acc_curves_dict[mintype][suffix] = [0] + suffixlst
    plot_learning_curve_full2(plot_acc_curves_dict, iterations, curve_save_path) 

    # Plot weight heatmap
    # heatmap_save_path = f"{WRITE_RESULT_FOLDER}/{FILE_PREFIX}_{VALIDATION_FILE_PREFIX}_{POOLING_FUNC_name}_HEATMAP_FINAL.jpg"  
    # heatmap_save_path = "LR-PL_heatmap_mindef.jpg"
    # get_heatmaps(model, POOLING_FUNC_name, heatmap_save_path)

    # Get accuracy on original training data
    # train_SGs, train_PLs, train_Ls = process_file(train_data_filepath)
    # X_train, y_train = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC)
    # y_pred = model.predict(X_train)
    # train_acc_score = accuracy_score(y_train, y_pred)
    # print(f"Train set acc: {train_acc_score}")
