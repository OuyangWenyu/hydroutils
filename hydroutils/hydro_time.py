"""
Author: Wenyu Ouyang
Date: 2022-12-02 11:03:04
LastEditTime: 2024-09-14 13:57:36
LastEditors: Wenyu Ouyang
Description: some functions to deal with time
FilePath: \\hydroutils\\hydroutils\\hydro_time.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import datetime
from typing import Union
import numpy as np
import pandas as pd
import pytz
import tzfpy


def t2str(t_: Union[str, datetime.datetime]):
    """Convert between datetime string and datetime object.

    Args:
        t_ (Union[str, datetime.datetime]): Input time, either as string or datetime object.

    Returns:
        Union[str, datetime.datetime]: If input is string, returns datetime object.
                                     If input is datetime, returns string.

    Raises:
        NotImplementedError: If input type is not supported.

    Note:
        String format is always "%Y-%m-%d".
    """
    if type(t_) is str:
        return datetime.datetime.strptime(t_, "%Y-%m-%d")
    elif type(t_) is datetime.datetime:
        return t_.strftime("%Y-%m-%d")
    else:
        raise NotImplementedError("We don't support this data type yet")


def t_range_days(t_range, *, step=np.timedelta64(1, "D")) -> np.array:
    """Transform a date range into a uniformly-spaced array of dates.

    Args:
        t_range (list): Two-element list containing start and end dates as strings.
        step (np.timedelta64, optional): Time interval between dates. Defaults to 1 day.

    Returns:
        np.array: Array of datetime64 objects with uniform spacing.

    Example:
        >>> t_range_days(["2000-01-01", "2000-01-05"])
        array(['2000-01-01', '2000-01-02', '2000-01-03', '2000-01-04'],
              dtype='datetime64[D]')
    """
    sd = datetime.datetime.strptime(t_range[0], "%Y-%m-%d")
    ed = datetime.datetime.strptime(t_range[1], "%Y-%m-%d")
    return np.arange(sd, ed, step)


def t_range_days_timedelta(t_array, td=12, td_type="h"):
    """Add a time delta to each date in an array.

    Args:
        t_array (np.array): Array of datetime64 objects (output of t_range_days).
        td (int, optional): Time period value. Defaults to 12.
        td_type (str, optional): Time period unit ('Y','M','D','h','m','s'). Defaults to "h".

    Returns:
        np.array: New array with time delta added to each element.

    Raises:
        AssertionError: If td_type is not one of 'Y','M','D','h','m','s'.
    """
    assert td_type in ["Y", "M", "D", "h", "m", "s"]
    t_array_final = [t + np.timedelta64(td, td_type) for t in t_array]
    return np.array(t_array_final)


def t_days_lst2range(t_array: list) -> list:
    """Transform a list of dates into a start-end interval.

    Args:
        t_array (list[Union[np.datetime64, str]]): List of dates in chronological order.

    Returns:
        list: Two-element list containing first and last dates as strings.

    Example:
        >>> t_days_lst2range(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"])
        ["2000-01-01", "2000-01-04"]
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
    """Get array of years covered by a date range.

    Args:
        t_range (list): Two-element list of dates as strings ["YYYY-MM-DD", "YYYY-MM-DD"].

    Returns:
        np.array: Array of years covered by the date range.

    Note:
        - Range is left-closed and right-open interval.
        - If end date is not January 1st, end year is included.
        - Example: ["2000-01-01", "2002-01-01"] -> [2000, 2001]
        - Example: ["2000-01-01", "2002-06-01"] -> [2000, 2001, 2002]
    """
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
    """Extract year from various time formats.

    Args:
        a_time (Union[datetime.date, np.datetime64, str]): Time in various formats.

    Returns:
        int: Year value.

    Note:
        Supports datetime.date, numpy.datetime64, and string formats.
        For strings, assumes YYYY is at the start.
    """
    if isinstance(a_time, datetime.date):
        return a_time.year
    elif isinstance(a_time, np.datetime64):
        return a_time.astype("datetime64[Y]").astype(int) + 1970
    else:
        return int(a_time[:4])


def intersect(t_lst1, t_lst2):
    """Find indices of common elements between two time lists.

    Args:
        t_lst1 (array-like): First time array.
        t_lst2 (array-like): Second time array.

    Returns:
        tuple: (ind1, ind2) where ind1 and ind2 are indices of common elements
        in t_lst1 and t_lst2 respectively.
    """
    C, ind1, ind2 = np.intersect1d(t_lst1, t_lst2, return_indices=True)
    return ind1, ind2


def date_to_julian(a_time):
    """Convert a date to Julian day of the year.

    Args:
        a_time (Union[str, datetime.datetime]): Date to convert.
            If string, must be in format 'YYYY-MM-DD'.

    Returns:
        int: Day of the year (1-366).
    """
    if type(a_time) == str:
        fmt = "%Y-%m-%d"
        dt = datetime.datetime.strptime(a_time, fmt)
    else:
        dt = a_time
    tt = dt.timetuple()
    return tt.tm_yday


def t_range_to_julian(t_range):
    """Convert a date range to a list of Julian days.

    Args:
        t_range (list): Two-element list of dates as strings ["YYYY-MM-DD", "YYYY-MM-DD"].

    Returns:
        list[int]: List of Julian days for each date in the range.
    """
    t_array = t_range_days(t_range)
    t_array_str = np.datetime_as_string(t_array)
    return [date_to_julian(a_time[:10]) for a_time in t_array_str]


def calculate_utc_offset(lat, lng, date=None):
    """Calculate the UTC offset for a geographic location.

    This function determines the timezone and UTC offset for a given latitude and
    longitude coordinate pair using the tzfpy library, which provides accurate
    timezone data based on geographic location.

    Args:
        lat (float): Latitude in decimal degrees (-90 to 90).
        lng (float): Longitude in decimal degrees (-180 to 180).
        date (datetime.datetime, optional): The date to calculate the offset for.
            Defaults to current UTC time. Important for handling daylight saving time.

    Returns:
        int: UTC offset in hours, or None if timezone cannot be determined.

    Example:
        >>> calculate_utc_offset(35.6762, 139.6503)  # Tokyo, Japan
        9
        >>> calculate_utc_offset(51.5074, -0.1278)   # London, UK
        0  # or 1 during DST
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
    """Generate a time range with annual reset to January 1st.

    This function creates a time range with a specified frequency, but with the special
    behavior that each year starts from January 1st regardless of the frequency interval.
    This is particularly useful for creating time series that need to align with
    calendar years while maintaining a regular interval pattern within each year.

    Args:
        start_time (Union[str, pd.Timestamp]): Start date of the range.
            Can be string ('YYYY-MM-DD') or pandas Timestamp.
        end_time (Union[str, pd.Timestamp]): End date of the range.
            Can be string ('YYYY-MM-DD') or pandas Timestamp.
        freq (str, optional): Time frequency for intervals. Defaults to '8D'.
            Common values: '7D' (weekly), '10D' (dekadal), etc.

    Returns:
        pd.DatetimeIndex: Time range with specified frequency and annual reset.

    Example:
        >>> generate_start0101_time_range('2020-03-15', '2021-02-15', freq='10D')
        DatetimeIndex(['2020-03-15', '2020-03-25', '2020-04-04', ...,
                      '2021-01-01', '2021-01-11', '2021-02-11'],
                      dtype='datetime64[ns]', freq=None)

    Note:
        - If an interval would cross into a new year, it's truncated and the next
          interval starts from January 1st of the new year.
        - The frequency must be a valid pandas frequency string that represents
          a fixed duration.
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
    """Determine start and end times from multiple time ranges.

    Args:
        time_ranges (list): List of time range pairs [[start1, end1], [start2, end2], ...].
            Each start/end can be any comparable type (datetime, string, etc.).
        assign_way (str, optional): Method to determine the final range. Defaults to "intersection".
            - "intersection": Use latest start time and earliest end time.
            - "union": Use earliest start time and latest end time.

    Returns:
        tuple: (time_start, time_end) The determined start and end times.

    Raises:
        NotImplementedError: If assign_way is not "intersection" or "union".

    Example:
        >>> ranges = [["2020-01-01", "2020-12-31"], ["2020-03-01", "2021-02-28"]]
        >>> assign_time_start_end(ranges, "intersection")
        ("2020-03-01", "2020-12-31")
        >>> assign_time_start_end(ranges, "union")
        ("2020-01-01", "2021-02-28")
    """
    if assign_way == "intersection":
        time_start = max(t[0] for t in time_ranges)
        time_end = min(t[1] for t in time_ranges)
    elif assign_way == "union":
        time_start = min(t[0] for t in time_ranges)
        time_end = max(t[1] for t in time_ranges)
    else:
        raise NotImplementedError("We don't support this assign_way yet")
    return time_start, time_end
