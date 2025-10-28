"""
Author: Wenyu Ouyang
Date: 2022-12-02 10:42:19
LastEditTime: 2025-10-28 08:16:56
LastEditors: Wenyu Ouyang
Description: Top-level package for hydroutils.
FilePath: \hydroutils\hydroutils\__init__.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

__author__ = """Wenyu Ouyang"""
__email__ = 'wenyuouyang@outlook.com'
__version__ = '0.1.0'

from .hydro_log import *
from .hydro_event import *
from .hydro_units import (
    streamflow_unit_conv,
    detect_time_interval,
    get_time_interval_info,
    validate_unit_compatibility,
)
from .hydro_units import *
from .hydro_file import *
from .hydro_stat import *
from .hydro_time import *
from .hydro_plot import *
from .hydro_correct import *