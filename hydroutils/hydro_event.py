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
    """
    将时间对象转换为十位数字格式 YYYYMMDDHH

    Parameters
    ----------
    time_obj : various
        时间对象，可以是 datetime, numpy.datetime64, 或字符串

    Returns
    -------
    str
        十位数字格式的时间字符串 YYYYMMDDHH
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
    """
    洪水事件提取函数 - 基于flood_event列提取

    这个函数的设计理念是：只要有flood_event列标记洪水事件，就可以提取事件，
    不需要关心其他列的具体名称，让调用者自己决定如何处理数据列。

    Parameters
    ----------
    df : pd.DataFrame
        包含站点数据的DataFrame，必须包含flood_event和time列
    warmup_length : int, default=0
        预热期长度（时间步数）
    flood_event_col : str, default="flood_event"
        洪水事件标记列名
    time_col : str, default="time"
        时间列名

    Returns
    -------
    List[Dict]
        洪水事件列表，每个字典包含：
        - event_name: 事件名称（基于时间）
        - start_idx: 实际事件开始索引（在原DataFrame中）
        - end_idx: 实际事件结束索引（在原DataFrame中）
        - warmup_start_idx: 预热期开始索引（在原DataFrame中）
        - data: 事件数据DataFrame（包含预热期）
        - is_warmup_mask: 布尔数组，标记哪些行是预热期（True=预热期，False=实际事件）
        - actual_start_time: 实际事件开始时间
        - actual_end_time: 实际事件结束时间
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
    """
    提取单个洪水事件的数据

    Parameters
    ----------
    df : pd.DataFrame
        包含时间序列数据的DataFrame
    start_idx : int
        实际洪水事件开始索引
    end_idx : int
        实际洪水事件结束索引
    warmup_length : int
        预热期长度（时间步数）
    flood_event_col : str, default="flood_event"
        洪水事件标记列名
    time_col : str, default="time"
        时间列名

    Returns
    -------
    Optional[Dict]
        如果事件有效，返回包含事件信息和数据的字典
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
    """
    仅获取洪水事件的索引信息，不提取数据

    Parameters
    ----------
    df : pd.DataFrame
        包含站点数据的DataFrame
    warmup_length : int, default=0
        预热期长度（时间步数）
    flood_event_col : str, default="flood_event"
        洪水事件标记列名

    Returns
    -------
    List[Dict]
        事件索引信息列表，每个字典包含：
        - start_idx: 实际事件开始索引
        - end_idx: 实际事件结束索引
        - warmup_start_idx: 预热期开始索引
        - duration: 实际事件持续时间步数
        - total_length: 包含预热期的总长度
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
    """
    根据事件索引和指定列名提取事件数据

    Parameters
    ----------
    df : pd.DataFrame
        原始DataFrame
    event_indices : Dict
        事件索引信息（来自get_event_indices或extract_flood_events_simple）
    data_columns : List[str]
        要提取的数据列名列表

    Returns
    -------
    Dict
        包含指定列数据的字典，键为列名，值为numpy数组
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
    """
    从洪水事件标记数组中找到连续的事件段

    这是一个底层函数，专门处理 flood_event 数组的分割逻辑，
    可以被不同的上层函数复用。

    Parameters
    ----------
    flood_event_array : np.ndarray
        洪水事件标记数组，>0 表示洪水事件
    warmup_length : int, default=0
        预热期长度（时间步数）

    Returns
    -------
    List[Dict]
        事件段信息列表，每个字典包含：
        - extended_start: 包含预热期的开始索引
        - extended_end: 事件结束索引
        - original_start: 实际事件开始索引
        - original_end: 实际事件结束索引
        - duration: 实际事件持续时间步数
        - total_length: 包含预热期的总长度
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
    """
    从洪水事件标记数组中找到连续的事件段（元组格式）

    这是为了兼容现有代码的便利函数。

    Parameters
    ----------
    flood_event_array : np.ndarray
        洪水事件标记数组，>0 表示洪水事件
    warmup_length : int, default=0
        预热期长度（时间步数）

    Returns
    -------
    List[Tuple[int, int, int, int]]
        (extended_start, extended_end, original_start, original_end) 元组列表
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
