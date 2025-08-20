"""
Author: Wenyu Ouyang
Date: 2025-01-17
LastEditTime: 2025-08-17 09:29:42
LastEditors: Wenyu Ouyang
Description: Flood event extraction utilities for hydrological data processing
FilePath: \hydromodeld:\Code\hydroutils\hydroutils\hydro_event.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple


def time_to_ten_digits(time_obj) -> str:
    """Convert a time object to a ten-digit format YYYYMMDDHH.

    Args:
        time_obj (Union[datetime.datetime, np.datetime64, str]): Time object to convert.
            Can be datetime, numpy.datetime64, or string.

    Returns:
        str: Ten-digit time string in YYYYMMDDHH format.

    Example:
        >>> time_to_ten_digits(datetime.datetime(2020, 1, 1, 12, 0))
        '2020010112'
        >>> time_to_ten_digits(np.datetime64('2020-01-01T12'))
        '2020010112'
        >>> time_to_ten_digits('2020-01-01T12:00:00')
        '2020010112'
    """
    if isinstance(time_obj, np.datetime64):
        # 如果是numpy datetime64对象
        return (
            time_obj.astype("datetime64[h]")
            .astype(str)
            .replace("-", "")
            .replace("T", "")
            .replace(":", "")
        )
    elif hasattr(time_obj, "strftime"):
        # 如果是datetime对象
        return time_obj.strftime("%Y%m%d%H")
    else:
        # 如果是字符串，尝试解析
        try:
            if isinstance(time_obj, str):
                dt = datetime.fromisoformat(time_obj.replace("Z", "+00:00"))
                return dt.strftime("%Y%m%d%H")
            else:
                return "0000000000"  # 默认值
        except Exception:
            return "0000000000"  # 默认值


def extract_flood_events(
    df: pd.DataFrame,
    warmup_length: int = 0,
    flood_event_col: str = "flood_event",
    time_col: str = "time",
) -> List[Dict]:
    """Extract flood events from a DataFrame based on a flood event indicator column.

    This function extracts flood events based on a binary indicator column (flood_event).
    The design philosophy is to be agnostic about other columns, letting the caller
    decide how to handle the data columns. The function only requires the flood_event
    column to mark events and a time column for event naming.

    Args:
        df (pd.DataFrame): DataFrame containing site data. Must have flood_event and
            time columns.
        warmup_length (int, optional): Number of time steps to include as warmup
            period before each event. Defaults to 0.
        flood_event_col (str, optional): Name of the flood event indicator column.
            Defaults to "flood_event".
        time_col (str, optional): Name of the time column. Defaults to "time".

    Returns:
        List[Dict]: List of flood events. Each dictionary contains:
            - event_name (str): Event name based on start/end times
            - start_idx (int): Start index of actual event in original DataFrame
            - end_idx (int): End index of actual event in original DataFrame
            - warmup_start_idx (int): Start index including warmup period
            - data (pd.DataFrame): Event data including warmup period
            - is_warmup_mask (np.ndarray): Boolean array marking warmup rows
            - actual_start_time: Start time of actual event
            - actual_end_time: End time of actual event

    Raises:
        ValueError: If required columns are missing from DataFrame.

    Example:
        >>> df = pd.DataFrame({
        ...     'time': pd.date_range('2020-01-01', periods=5),
        ...     'flood_event': [0, 1, 1, 1, 0],
        ...     'flow': [100, 200, 300, 250, 150]
        ... })
        >>> events = extract_flood_events(df, warmup_length=1)
        >>> len(events)
        1
        >>> events[0]['data']
           time  flood_event  flow
        0  2020-01-01    0  100  # warmup period
        1  2020-01-02    1  200  # event start
        2  2020-01-03    1  300
        3  2020-01-04    1  250  # event end
    """
    events: List[Dict] = []

    # 检查必要的列是否存在
    required_cols = [flood_event_col, time_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要的列: {missing_cols}")

    # 找到连续的flood_event > 0区间
    flood_mask = df[flood_event_col] > 0
    if not flood_mask.any():
        return events

    # 找连续区间
    in_event = False
    start_idx = None

    for idx, is_flood in enumerate(flood_mask):
        if is_flood and not in_event:
            start_idx = idx
            in_event = True
        elif not is_flood and in_event and start_idx is not None:
            # 事件结束，提取事件数据
            event_dict = _extract_single_event(
                df, start_idx, idx, warmup_length, flood_event_col, time_col
            )
            if event_dict is not None:
                events.append(event_dict)
            in_event = False

    # 处理最后一个事件（如果数据结束时仍在事件中）
    if in_event and start_idx is not None:
        event_dict = _extract_single_event(
            df, start_idx, len(df), warmup_length, flood_event_col, time_col
        )
        if event_dict is not None:
            events.append(event_dict)

    return events


def _extract_single_event(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    warmup_length: int,
    flood_event_col: str = "flood_event",
    time_col: str = "time",
) -> Optional[Dict]:
    """Extract data for a single flood event.

    This internal function handles the extraction of data for a single flood event,
    including the warmup period. It creates a dictionary containing all relevant
    information about the event.

    Args:
        df (pd.DataFrame): DataFrame containing time series data.
        start_idx (int): Start index of the actual flood event.
        end_idx (int): End index of the actual flood event.
        warmup_length (int): Number of time steps to include as warmup period.
        flood_event_col (str, optional): Name of flood event indicator column.
            Defaults to "flood_event".
        time_col (str, optional): Name of time column. Defaults to "time".

    Returns:
        Optional[Dict]: If event is valid, returns a dictionary containing:
            - event_name (str): Event name based on start/end times
            - start_idx (int): Start index of actual event
            - end_idx (int): End index of actual event
            - warmup_start_idx (int): Start index including warmup
            - data (pd.DataFrame): Event data including warmup
            - is_warmup_mask (np.ndarray): Boolean array marking warmup rows
            - actual_start_time: Start time of actual event
            - actual_end_time: End time of actual event
            Returns None if event data is invalid (less than 2 rows).
    """
    warmup_start_idx = max(0, start_idx - warmup_length)

    # 提取包含预热期的数据
    event_data = df.iloc[warmup_start_idx:end_idx].copy()

    # 基本验证 - 检查是否有有效数据
    if len(event_data) < 2:
        return None

    # 获取实际洪水事件的开始和结束时间
    actual_start_time = df.iloc[start_idx][time_col]
    actual_end_time = df.iloc[end_idx - 1][time_col]

    # 生成事件名称
    start_digits = time_to_ten_digits(actual_start_time)
    end_digits = time_to_ten_digits(actual_end_time)
    event_name = f"{start_digits}_{end_digits}"

    # 创建预热期标记数组
    is_warmup_mask = np.zeros(len(event_data), dtype=bool)
    warmup_rows = start_idx - warmup_start_idx
    if warmup_rows > 0:
        is_warmup_mask[:warmup_rows] = True

    return {
        "event_name": event_name,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "warmup_start_idx": warmup_start_idx,
        "data": event_data,
        "is_warmup_mask": is_warmup_mask,
        "actual_start_time": actual_start_time,
        "actual_end_time": actual_end_time,
    }


def get_event_indices(
    df: pd.DataFrame, warmup_length: int = 0, flood_event_col: str = "flood_event"
) -> List[Dict]:
    """Get index information for flood events without extracting data.

    This function identifies flood events in the DataFrame and returns their index
    information, but does not extract the actual data. This is useful when you only
    need to know the locations and durations of events.

    Args:
        df (pd.DataFrame): DataFrame containing site data.
        warmup_length (int, optional): Number of time steps to include as warmup
            period before each event. Defaults to 0.
        flood_event_col (str, optional): Name of flood event indicator column.
            Defaults to "flood_event".

    Returns:
        List[Dict]: List of event index information. Each dictionary contains:
            - start_idx (int): Start index of actual event
            - end_idx (int): End index of actual event
            - warmup_start_idx (int): Start index including warmup period
            - duration (int): Duration of actual event in time steps
            - total_length (int): Total length including warmup period

    Raises:
        ValueError: If flood_event_col is not found in DataFrame.

    Example:
        >>> df = pd.DataFrame({'flood_event': [0, 1, 1, 1, 0]})
        >>> indices = get_event_indices(df, warmup_length=1)
        >>> indices[0]
        {
            'start_idx': 1,
            'end_idx': 4,
            'warmup_start_idx': 0,
            'duration': 3,
            'total_length': 4
        }
    """
    # 检查必要的列是否存在
    if flood_event_col not in df.columns:
        raise ValueError(f"DataFrame缺少洪水事件标记列: {flood_event_col}")

    # 使用底层函数处理分割逻辑
    flood_event_array = df[flood_event_col].values
    segments = find_flood_event_segments_from_array(flood_event_array, warmup_length)

    # 转换为与原接口兼容的格式
    events = []
    for seg in segments:
        events.append(
            {
                "start_idx": seg["original_start"],
                "end_idx": seg["original_end"] + 1,  # +1 因为原来是不包含结束索引的
                "warmup_start_idx": seg["extended_start"],
                "duration": seg["duration"],
                "total_length": seg["total_length"],
            }
        )

    return events


def extract_event_data_by_columns(
    df: pd.DataFrame, event_indices: Dict, data_columns: List[str]
) -> Dict:
    """Extract event data for specified columns using event indices.

    This function extracts data from specified columns for a flood event using
    the index information from get_event_indices or extract_flood_events.

    Args:
        df (pd.DataFrame): Original DataFrame containing all data.
        event_indices (Dict): Event index information dictionary containing:
            - warmup_start_idx (int): Start index including warmup period
            - end_idx (int): End index of event
        data_columns (List[str]): List of column names to extract.

    Returns:
        Dict: Dictionary mapping column names to numpy arrays containing the
            extracted data. If a column is not found, it will contain an array
            of NaN values.

    Example:
        >>> df = pd.DataFrame({
        ...     'time': pd.date_range('2020-01-01', periods=5),
        ...     'flow': [100, 200, 300, 250, 150]
        ... })
        >>> indices = {'warmup_start_idx': 1, 'end_idx': 4}
        >>> data = extract_event_data_by_columns(df, indices, ['flow'])
        >>> data['flow']
        array([200., 300., 250.])
    """
    start_idx = event_indices["warmup_start_idx"]
    end_idx = event_indices["end_idx"]

    event_data = {}
    for col in data_columns:
        if col in df.columns:
            event_data[col] = df.iloc[start_idx:end_idx][col].values
        else:
            # 如果列不存在，用NaN数组填充
            event_data[col] = np.full(end_idx - start_idx, np.nan)

    return event_data


def find_flood_event_segments_from_array(
    flood_event_array: np.ndarray,
    warmup_length: int = 0,
) -> List[Dict]:
    """Find continuous flood event segments in a binary indicator array.

    This is a low-level function that handles the core logic of segmenting a
    flood event indicator array into continuous events. It can be reused by
    different higher-level functions.

    Args:
        flood_event_array (np.ndarray): Binary array where values > 0 indicate
            flood events.
        warmup_length (int, optional): Number of time steps to include as warmup
            period before each event. Defaults to 0.

    Returns:
        List[Dict]: List of event segment information. Each dictionary contains:
            - extended_start (int): Start index including warmup period
            - extended_end (int): End index of event
            - original_start (int): Start index of actual event
            - original_end (int): End index of actual event
            - duration (int): Duration of actual event in time steps
            - total_length (int): Total length including warmup period

    Example:
        >>> arr = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        >>> segments = find_flood_event_segments_from_array(arr, warmup_length=1)
        >>> len(segments)  # Two events found
        2
        >>> segments[0]  # First event with one timestep warmup
        {
            'extended_start': 1,  # Warmup start
            'extended_end': 4,    # Event end
            'original_start': 2,  # Actual event start
            'original_end': 4,    # Actual event end
            'duration': 3,        # Event duration
            'total_length': 4     # Total length with warmup
        }
    """
    segments = []

    # 找到所有 flood_event > 0 的索引
    event_indices = np.where(flood_event_array > 0)[0]

    if len(event_indices) == 0:
        return segments

    # 找到连续段的分割点
    gaps = np.diff(event_indices) > 1
    split_points = np.where(gaps)[0] + 1
    split_indices = np.split(event_indices, split_points)

    # 为每个连续段生成信息
    for indices in split_indices:
        if len(indices) > 0:
            original_start = indices[0]
            original_end = indices[-1]

            # 添加预热期
            extended_start = max(0, original_start - warmup_length)

            segments.append(
                {
                    "extended_start": extended_start,
                    "extended_end": original_end,
                    "original_start": original_start,
                    "original_end": original_end,
                    "duration": original_end - original_start + 1,
                    "total_length": original_end - extended_start + 1,
                }
            )

    return segments


def find_flood_event_segments_as_tuples(
    flood_event_array: np.ndarray,
    warmup_length: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Find continuous flood event segments and return them as tuples.

    This is a convenience function that returns event segments as tuples instead
    of dictionaries, for compatibility with existing code.

    Args:
        flood_event_array (np.ndarray): Binary array where values > 0 indicate
            flood events.
        warmup_length (int, optional): Number of time steps to include as warmup
            period before each event. Defaults to 0.

    Returns:
        List[Tuple[int, int, int, int]]: List of tuples, each containing:
            (extended_start, extended_end, original_start, original_end)
            where:
            - extended_start: Start index including warmup period
            - extended_end: End index of event
            - original_start: Start index of actual event
            - original_end: End index of actual event

    Example:
        >>> arr = np.array([0, 0, 1, 1, 1, 0])
        >>> segments = find_flood_event_segments_as_tuples(arr, warmup_length=1)
        >>> segments[0]  # (warmup_start, event_end, event_start, event_end)
        (1, 4, 2, 4)
    """
    segments = find_flood_event_segments_from_array(flood_event_array, warmup_length)

    return [
        (
            seg["extended_start"],
            seg["extended_end"],
            seg["original_start"],
            seg["original_end"],
        )
        for seg in segments
    ]
