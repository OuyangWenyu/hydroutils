"""
Author: Wenyu Ouyang
Date: 2022-12-02 11:03:04
LastEditTime: 2024-09-14 13:57:36
LastEditors: Wenyu Ouyang
Description: some functions to deal with time
FilePath: \hydroutils\hydroutils\hydro_time.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import datetime
from typing import Union
import numpy as np
import pandas as pd
import pytz
import tzfpy


def t2str(t_: Union[str, datetime.datetime]):
    if type(t_) is str:
        return datetime.datetime.strptime(t_, "%Y-%m-%d")
    elif type(t_) is datetime.datetime:
        return t_.strftime("%Y-%m-%d")
    else:
        raise NotImplementedError("We don't support this data type yet")


def t_range_days(t_range, *, step=np.timedelta64(1, "D")) -> np.array:
    """
    Transform the two-value t_range list to a uniformly-spaced list (default is a daily list).
    For example, ["2000-01-01", "2000-01-05"] -> ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]
    Parameters
    ----------
    t_range
        two-value t_range list
    step
        the time interval; its default value is 1 day
    Returns
    -------
    np.array
        a uniformly-spaced (daily) list
    """
    sd = datetime.datetime.strptime(t_range[0], "%Y-%m-%d")
    ed = datetime.datetime.strptime(t_range[1], "%Y-%m-%d")
    return np.arange(sd, ed, step)


def t_range_days_timedelta(t_array, td=12, td_type="h"):
    """
    for each day, add a timedelta
    Parameters
    ----------
    t_array
        its data type is same as the return type of "t_range_days" function
    td
        time periods
    td_type
        the type of time period
    Returns
    -------
    np.array
        a new t_array
    """
    assert td_type in ["Y", "M", "D", "h", "m", "s"]
    t_array_final = [t + np.timedelta64(td, td_type) for t in t_array]
    return np.array(t_array_final)


def t_days_lst2range(t_array: list) -> list:
    """
    Transform a period list to its interval.
    For example,  ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"] ->  ["2000-01-01", "2000-01-04"]
    Parameters
    ----------
    t_array: list[Union[np.datetime64, str]]
        a period list
    Returns
    -------
    list
        An time interval
    """
    if type(t_array[0]) == np.datetime64:
        t0 = t_array[0].astype(datetime.datetime)
        t1 = t_array[-1].astype(datetime.datetime)
    else:
        t0 = t_array[0]
        t1 = t_array[-1]
    sd = t0.strftime("%Y-%m-%d")
    ed = t1.strftime("%Y-%m-%d")
    return [sd, ed]


def t_range_years(t_range):
    """t_range is a left-closed and right-open interval, if t_range[1] is not Jan.1 then end_year should be included"""
    start_year = int(t_range[0].split("-")[0])
    end_year = int(t_range[1].split("-")[0])
    end_month = int(t_range[1].split("-")[1])
    end_day = int(t_range[1].split("-")[2])
    return (
        np.arange(start_year, end_year)
        if end_month == 1 and end_day == 1
        else np.arange(start_year, end_year + 1)
    )


def get_year(a_time):
    if isinstance(a_time, datetime.date):
        return a_time.year
    elif isinstance(a_time, np.datetime64):
        return a_time.astype("datetime64[Y]").astype(int) + 1970
    else:
        return int(a_time[:4])


def intersect(t_lst1, t_lst2):
    C, ind1, ind2 = np.intersect1d(t_lst1, t_lst2, return_indices=True)
    return ind1, ind2


def date_to_julian(a_time):
    if type(a_time) == str:
        fmt = "%Y-%m-%d"
        dt = datetime.datetime.strptime(a_time, fmt)
    else:
        dt = a_time
    tt = dt.timetuple()
    return tt.tm_yday


def t_range_to_julian(t_range):
    t_array = t_range_days(t_range)
    t_array_str = np.datetime_as_string(t_array)
    return [date_to_julian(a_time[:10]) for a_time in t_array_str]


def calculate_utc_offset(lat, lng, date=None):
    """
    Calculate the UTC offset for a given latitude and longitude using tzfpy.

    Parameters
    ----------
    lat : float
        Latitude.
    lng : float
        Longitude.
    date : datetime, optional
        The date to consider for the UTC offset. If not provided, uses the current date.

    Returns
    -------
    int
        UTC offset in hours.
    """
    if date is None:
        date = datetime.datetime.utcnow()

    if timezone_str := tzfpy.get_tz(lng, lat):
        # Get the timezone object using pytz
        tz = pytz.timezone(timezone_str)
        # Get the UTC offset for the specified date
        offset = tz.utcoffset(date)
        if offset is not None:
            return int(offset.total_seconds() / 3600)
    return None


def generate_start0101_time_range(start_time, end_time, freq="8D"):
    """Generate a time range with a flexible start date and each year starting from 01-01.

    Parameters
    ----------
    start_time : str or pd.Timestamp
        The start time for the range (could be any date, string or Timestamp).
    end_time : str or pd.Timestamp
        The end time for the range (could be any date, string or Timestamp).
    freq : str, optional
        Time frequency for intervals, by default '8D'. Could be '7D', '10D', etc.

    Returns
    -------
    pd.DatetimeIndex
        A time range index with custom intervals and annual reset at 01-01.
    """
    all_dates = []

    # Ensure the start and end times are of type pd.Timestamp
    current_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # Parse the frequency interval correctly
    interval_days = pd.Timedelta(freq)  # Ensure it's a Timedelta

    while current_time <= end_time:
        all_dates.append(current_time)

        # Calculate next date with the specified interval
        next_time = current_time + interval_days

        # If next_time crosses into a new year, reset to 01-01 of the new year
        if next_time.year > current_time.year:
            next_time = pd.Timestamp(f"{next_time.year}-01-01")

        current_time = next_time

    return pd.to_datetime(all_dates)


def assign_time_start_end(time_ranges, assign_way="intersection"):
    # chose all time start from time_ranges:[[start1, end1], [start2, end2], ...]
    if assign_way == "intersection":
        time_start = max(t[0] for t in time_ranges)
        time_end = min(t[1] for t in time_ranges)
    elif assign_way == "union":
        time_start = min(t[0] for t in time_ranges)
        time_end = max(t[1] for t in time_ranges)
    else:
        raise NotImplementedError("We don't support this assign_way yet")
    return time_start, time_end
