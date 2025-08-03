"""
Author: Wenyu Ouyang
Date: 2023-07-29 14:07:03
LastEditTime: 2023-07-29 14:08:22
LastEditors: Wenyu Ouyang
Description: some common arithmetric calculations
FilePath: \\hydroutils\\hydroutils\\hydro_arithmetric.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import numpy as np


def random_choice_no_return(arr, num_lst):
    """sampling without replacement multi-times, and the num of each time is in num_lst"""
    num_lst_arr = np.array(num_lst)
    num_sum = num_lst_arr.sum()
    if type(arr) == list:
        arr = np.array(arr)
    assert num_sum <= arr.size
    results = []
    arr_residue = np.arange(arr.size)
    for num in num_lst_arr:
        idx_chosen = np.random.choice(arr_residue.size, num, replace=False)
        chosen_idx_in_arr = np.sort(arr_residue[idx_chosen])
        results.append(arr[chosen_idx_in_arr])
        arr_residue = np.delete(arr_residue, idx_chosen)
    return results
