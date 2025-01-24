"""Tools and scripts for analyzing output data from models."""

from typing import List, Dict

"""
Temporary data structure that tracks performance of each stimuli set + suffix class combination. Will typically be later parsed into acc_dict data type to pass into plotting functions.

epoch_data_dict = {
    0: {
        "minDefault_suffixA": [acc1, acc2, acc3, ...],
        "minDefault_suffixB": [acc1, acc2, acc3, ...],
        ...
        "majDeault_suffixC": [acc2, acc2, acc3, ...]
    }
    1: {
        "minDefault_suffixA": [acc1, acc2, acc3, ...],
        "minDefault_suffixB": [acc1, acc2, acc3, ...],
        ...
        "majDeault_suffixC": [acc2, acc2, acc3, ...]
    }
    ...
}
"""
epoch_data_dict = Dict[int, Dict[str, List[float]]]

"""
Data structure that maps the accuracies across epochs for each stim_set and suffix class combination.

acc_dict = {
    "equalFreq": {
        "Suffix A": [acc1, acc2, acc3, ...],
        "Suffix B": [acc1, acc2, acc3, ...],
        "Suffix C": [acc1, acc2, acc3, ...]
    },
    "minDefault": {
        "Suffix A": [acc1, acc2, acc3, ...],
        "Suffix B": [acc1, acc2, acc3, ...],
        "Suffix C": [acc1, acc2, acc3, ...]
    },
    "majDefault": {
        "Suffix A": [acc1, acc2, acc3, ...],
        "Suffix B": [acc1, acc2, acc3, ...],
        "Suffix C": [acc1, acc2, acc3, ...]
    }
}
"""
acc_dict = Dict[str, Dict[str, List[float]]]