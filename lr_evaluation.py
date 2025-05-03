import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from helper import * 
from tqdm import tqdm
from pooling_functions import *
from sklearn.linear_model import SGDClassifier


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

            X_train_org, y_train_org = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC)

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
    curve_save_path = f"LR-PL_curve_all.jpg"
    num_batches = int(np.ceil(X_train.shape[0]/BATCH_SIZE))
    iterations = [i for i in range(len(avg_class_1_accs))] 
    plot_learning_curve_full(acc_curves_dict, iterations, curve_save_path) 


    # Get accuracy on original training data
    # train_SGs, train_PLs, train_Ls = process_file(train_data_filepath)
    # X_train, y_train = get_arrays(train_SGs, train_PLs, train_Ls, symbol2feats, suffix2label, pool_func=POOLING_FUNC)
    # y_pred = model.predict(X_train)
    # train_acc_score = accuracy_score(y_train, y_pred)
    # print(f"Train set acc: {train_acc_score}")
