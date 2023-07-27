#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2022-12-02 10:42:19
LastEditTime: 2023-07-27 10:02:57
LastEditors: Wenyu Ouyang
Description: Tests for `hydroutils` package
FilePath: \hydroutils\tests\test_hydroutils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import pytest
from hydroutils import hydro_time
import numpy as np


def test_time_func():
    t_range = ["2000-01-01", "2005-01-01"]
    time_range_lst = hydro_time.t_range_days(t_range)
    assert len(time_range_lst) == 1827
    new_t_range_lst = hydro_time.t_range_days_timedelta(time_range_lst)
    assert new_t_range_lst[0] == np.datetime64("2000-01-01T12:00")
