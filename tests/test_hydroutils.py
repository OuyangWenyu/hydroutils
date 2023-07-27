#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2022-12-02 10:42:19
LastEditTime: 2023-07-27 11:24:30
LastEditors: Wenyu Ouyang
Description: Tests for `hydroutils` package
FilePath: \hydroutils\tests\test_hydroutils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import pytest
from hydroutils import hydro_time
from hydroutils import hydro_stat
import numpy as np


def test_time_func():
    t_range = ["2000-01-01", "2005-01-01"]
    time_range_lst = hydro_time.t_range_days(t_range)
    assert len(time_range_lst) == 1827
    new_t_range_lst = hydro_time.t_range_days_timedelta(time_range_lst)
    assert new_t_range_lst[0] == np.datetime64("2000-01-01T12:00")


def test_stat_func():
    targ = np.full((3, 6), np.nan)
    targ[0, 0] = 1
    targ[1, 1] = 4
    targ[2, 2] = 7
    targ[0, 3] = 2
    targ[1, 4] = 5
    targ[2, 5] = 8
    pred = np.full((3, 6), 1)
    inds_sum = hydro_stat.stat_error(targ, pred, fill_nan="sum")
    inds_mean = hydro_stat.stat_error(targ, pred, fill_nan="mean")
    assert inds_sum["Bias"][0] == - 0.5
    assert inds_mean["RMSE"][0] == np.sqrt(1 / 2)
