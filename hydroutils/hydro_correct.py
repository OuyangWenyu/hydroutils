"""
Author: Wenyu Ouyang
Date: 2025-01-17
LastEditTime: 2025-10-28 08:22:40
LastEditors: Wenyu Ouyang
Description:
FilePath: \hydroutils\hydroutils\hydro_correct.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple


class HydrographCorrector:
    """Interactive corrector for flood hydrograph.

    This class implements a combination of five-point quadratic smoothing and cubic spline
    interpolation based on Zhang Silong's 2006 paper "Research on Interactive Correction
    Technology for Flood Forecasting Process".

    The corrector provides methods for:
        - Five-point quadratic smoothing
        - Cubic spline interpolation
        - Combined correction process

    Attributes:
        time_points (np.ndarray): Array of time points.
        discharge_original (np.ndarray): Original discharge values.
        n_points (int): Number of data points.
        smoothing_matrix (sparse.csr_matrix): Pre-computed smoothing matrix.
    """

    def __init__(self, time_points: np.ndarray, discharge_values: np.ndarray):
        """Initialize the hydrograph corrector.

        Args:
            time_points (np.ndarray): Array of time points.
            discharge_values (np.ndarray): Array of discharge values.

        Note:
            The smoothing matrix is pre-computed during initialization for efficiency.
            This matrix is used in the five-point quadratic smoothing process.
        """
        self.time_points = np.asarray(time_points)
        self.discharge_original = np.asarray(discharge_values)
        self.n_points = len(discharge_values)

        # 预计算平滑矩阵（只需计算一次）
        self.smoothing_matrix = self._create_smoothing_matrix()

    def _create_smoothing_matrix(self) -> sparse.csr_matrix:
        """Create five-point quadratic smoothing matrix.

        Constructs a sparse matrix for five-point quadratic smoothing based on equations
        (5)-(7) from the paper. Each row corresponds to smoothing coefficients for an
        output point.

        The matrix implements these formulas:
            - Middle points (eq. 5): [-3(y_{i-2} + y_{i+2}) + 12(y_{i-1} + y_{i+1}) + 17y_i] / 35
            - First two points (eq. 6):
                y_{-2} = (31y_{-2} + 9y_{-1} - 3y_0 - 5y_1 + 3y_2) / 35
                y_{-1} = (9y_{-2} + 13y_{-1} + 12y_0 + 6y_1 - 5y_2) / 35
            - Last two points (eq. 7):
                y_1 = (-5y_{-2} + 6y_{-1} + 12y_0 + 13y_1 + 9y_2) / 35
                y_2 = (3y_{-2} - 5y_{-1} - 3y_0 + 9y_1 + 31y_2) / 35

        Returns:
            sparse.csr_matrix: Sparse smoothing matrix (n x n). For n < 5, returns identity matrix.
        """
        n = self.n_points
        if n < 5:
            # 对于少于5个点的数据，返回单位矩阵（不进行平滑）
            return sparse.eye(n, format="csr")

        # 构造稀疏矩阵的行、列、数据
        rows, cols, data = [], [], []

        # 第一个点 (索引0) - 公式(6)第一个
        # y_{-2} = (31y_{-2} + 9y_{-1} - 3y_0 - 5y_1 + 3y_2) / 35
        coeffs_0 = np.array([31, 9, -3, -5, 3]) / 35
        for j, coeff in enumerate(coeffs_0):
            rows.append(0)
            cols.append(j)
            data.append(coeff)

        # 第二个点 (索引1) - 公式(6)第二个
        # y_{-1} = (9y_{-2} + 13y_{-1} + 12y_0 + 6y_1 - 5y_2) / 35
        coeffs_1 = np.array([9, 13, 12, 6, -5]) / 35
        for j, coeff in enumerate(coeffs_1):
            rows.append(1)
            cols.append(j)
            data.append(coeff)

        # 中间点 (索引2到n-3) - 公式(5)
        # y_0 = [-3(y_{-2} + y_2) + 12(y_{-1} + y_1) + 17y_0] / 35
        coeffs_mid = np.array([-3, 12, 17, 12, -3]) / 35
        for i in range(2, n - 2):
            for j, coeff in enumerate(coeffs_mid):
                rows.append(i)
                cols.append(i - 2 + j)
                data.append(coeff)

        # 倒数第二个点 (索引n-2) - 公式(7)第一个
        # y_1 = (-5y_{-2} + 6y_{-1} + 12y_0 + 13y_1 + 9y_2) / 35
        coeffs_n2 = np.array([-5, 6, 12, 13, 9]) / 35
        for j, coeff in enumerate(coeffs_n2):
            rows.append(n - 2)
            cols.append(n - 5 + j)
            data.append(coeff)

        # 最后一个点 (索引n-1) - 公式(7)第二个
        # y_2 = (3y_{-2} - 5y_{-1} - 3y_0 + 9y_1 + 31y_2) / 35
        coeffs_n1 = np.array([3, -5, -3, 9, 31]) / 35
        for j, coeff in enumerate(coeffs_n1):
            rows.append(n - 1)
            cols.append(n - 5 + j)
            data.append(coeff)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def five_point_smooth(self, discharge_values: np.ndarray) -> np.ndarray:
        """
        对径流数据进行五点二次平滑

        Args:
            discharge_values: 径流值数组

        Returns:
            平滑后的径流值数组
        """
        discharge_values = np.asarray(discharge_values)
        if len(discharge_values) != self.n_points:
            msg = f"输入数据长度 {len(discharge_values)} 与初始化长度 {self.n_points} 不匹配"
            raise ValueError(msg)

        # 使用预计算的稀疏矩阵进行向量化平滑
        smoothed = self.smoothing_matrix @ discharge_values

        # 确保结果非负（径流不能为负）
        return np.maximum(smoothed, 0.0)

    def _build_spline_system(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_prime_start: float = 0.0,
        y_prime_end: float = 0.0,
    ) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Build linear system for cubic spline interpolation.

        Constructs tridiagonal system A·M = β based on equations (17)-(21) from the paper.
        The system solves for second derivatives at each node point.

        Args:
            x (np.ndarray): Array of node positions.
            y (np.ndarray): Array of node values.
            y_prime_start (float, optional): First derivative at start point. Defaults to 0.0.
            y_prime_end (float, optional): First derivative at end point. Defaults to 0.0.

        Returns:
            Tuple[sparse.csr_matrix, np.ndarray]: A tuple containing:
                - Coefficient matrix A (sparse tridiagonal matrix)
                - Right-hand side vector β

        Note:
            The system uses natural spline boundary conditions by default (M₀ = Mₙ = 0).
            Internal points follow equation (17): αᵢMᵢ₋₁ + 2Mᵢ + (1-αᵢ)Mᵢ₊₁ = βᵢ
            where αᵢ = hᵢ/(hᵢ₋₁ + hᵢ)
        """
        n = len(x) - 1  # 区间数
        h = np.diff(x)  # h[i] = x[i+1] - x[i]

        # 构造三对角矩阵 A
        main_diag = np.ones(n + 1) * 2.0

        # 构造上下对角线
        upper_diag = np.zeros(n)  # 长度为n的上对角线
        lower_diag = np.zeros(n)  # 长度为n的下对角线

        # 边界条件
        if n > 0:
            upper_diag[0] = 1.0  # 第一行的边界条件
            lower_diag[-1] = 1.0  # 最后一行的边界条件

        # 内节点的α系数
        if n > 1:
            alpha = h[1:] / (h[:-1] + h[1:])  # α_i = h_{i+1}/(h_i + h_{i+1})

            # 填充内节点的系数
            for i in range(len(alpha)):
                if i + 1 < len(upper_diag):
                    upper_diag[i + 1] = alpha[i]
                if i < len(lower_diag) - 1:
                    lower_diag[i] = 1 - alpha[i]

        # 构造稀疏三对角矩阵
        diagonals = [lower_diag, main_diag, upper_diag]
        offsets = [-1, 0, 1]
        A = sparse.diags(diagonals, offsets, shape=(n + 1, n + 1), format="csr")

        # 构造右端向量β
        beta = np.zeros(n + 1)

        # 边界条件 - 公式(19)(20)
        if n > 0:
            beta[0] = 6.0 / h[0] * ((y[1] - y[0]) / h[0] - y_prime_start)
            beta[-1] = 6.0 / h[-1] * (y_prime_end - (y[-1] - y[-2]) / h[-1])

        # 内节点的β值 - 公式(17)
        for i in range(1, n):
            beta[i] = (
                6.0
                / (h[i - 1] + h[i])
                * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
            )

        return A, beta

    def _solve_spline_coefficients(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_prime_start: float = 0.0,
        y_prime_end: float = 0.0,
    ) -> np.ndarray:
        """Solve for second derivative coefficients of cubic spline interpolation.

        Args:
            x (np.ndarray): Node positions.
            y (np.ndarray): Node values.
            y_prime_start (float, optional): First derivative at start point. Defaults to 0.0.
            y_prime_end (float, optional): First derivative at end point. Defaults to 0.0.

        Returns:
            np.ndarray: Array of second derivatives (M values) at each node point.

        Note:
            Uses sparse matrix solver for the tridiagonal system.
            The solution provides the M values needed for cubic spline construction.
        """
        A, beta = self._build_spline_system(x, y, y_prime_start, y_prime_end)

        # 使用稀疏矩阵求解器（追赶法的高效实现）
        M = spsolve(A, beta)

        return M

    def _evaluate_spline(
        self, x_nodes: np.ndarray, y_nodes: np.ndarray, M: np.ndarray, x_new: np.ndarray
    ) -> np.ndarray:
        """Evaluate cubic spline function at new points.

        Vectorized implementation of equation (15) from the paper for efficient
        computation of spline values at multiple points.

        Args:
            x_nodes (np.ndarray): Original node positions.
            y_nodes (np.ndarray): Original node values.
            M (np.ndarray): Second derivatives at nodes.
            x_new (np.ndarray): New points at which to evaluate the spline.

        Returns:
            np.ndarray: Interpolated values at x_new points.

        Note:
            The implementation uses vectorized operations for efficiency.
            For each point, it:
            1. Finds the containing interval
            2. Computes local coordinates
            3. Evaluates the cubic spline formula
        """
        h = np.diff(x_nodes)
        n = len(x_nodes) - 1

        # 找到每个新点所在的区间
        indices = np.searchsorted(x_nodes[1:], x_new)
        indices = np.clip(indices, 0, n - 1)

        # 向量化计算 - 公式(15)
        x_left = x_nodes[indices]
        x_right = x_nodes[indices + 1]
        y_left = y_nodes[indices]
        y_right = y_nodes[indices + 1]
        M_left = M[indices]
        M_right = M[indices + 1]
        h_seg = h[indices]

        # 样条公式的向量化实现
        t1 = (x_right - x_new) / h_seg
        t2 = (x_new - x_left) / h_seg

        result = (
            (t1**3 * M_left + t2**3 * M_right) * h_seg**2 / 6.0
            + (y_left - M_left * h_seg**2 / 6.0) * t1
            + (y_right - M_right * h_seg**2 / 6.0) * t2
        )

        return result

    def cubic_spline_interpolation(
        self, x: np.ndarray, y: np.ndarray, x_new: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Perform cubic spline interpolation.

        Interpolates the given data points using cubic splines with natural boundary
        conditions. For sequences shorter than 3 points, falls back to linear interpolation.

        Args:
            x (np.ndarray): Original time points.
            y (np.ndarray): Original discharge values.
            x_new (Optional[np.ndarray], optional): New time points for interpolation.
                If None, uses original points. Defaults to None.

        Returns:
            np.ndarray: Interpolated discharge values.

        Note:
            Uses natural spline boundary conditions (zero second derivative at endpoints).
            For n < 3 points, automatically switches to linear interpolation.
            Ensures non-negative results for physical consistency.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if x_new is None:
            x_new = x
        else:
            x_new = np.asarray(x_new)

        if len(x) < 3:
            # 对于少于3个点的数据，使用线性插值
            from scipy.interpolate import interp1d

            f = interp1d(x, y, kind="linear", fill_value="extrapolate")
            return f(x_new)

        # 设置边界条件（自然边界：二阶导数为0）
        y_prime_start = 0.0
        y_prime_end = 0.0

        # 求解样条系数
        M = self._solve_spline_coefficients(x, y, y_prime_start, y_prime_end)

        # 计算插值结果
        result = self._evaluate_spline(x, y, M, x_new)

        # 确保结果非负
        return np.maximum(result, 0.0)

    def apply_correction(
        self,
        modified_discharge: np.ndarray,
        smoothing_enabled: bool = True,
        interpolation_enabled: bool = True,
    ) -> np.ndarray:
        """Apply the complete correction algorithm.

        Applies a combination of five-point quadratic smoothing and cubic spline
        interpolation to the discharge data. Both steps can be enabled/disabled
        independently.

        Args:
            modified_discharge (np.ndarray): Modified discharge data.
            smoothing_enabled (bool, optional): Whether to enable five-point smoothing.
                Defaults to True.
            interpolation_enabled (bool, optional): Whether to enable spline interpolation.
                Defaults to True.

        Returns:
            np.ndarray: Corrected discharge data.

        Note:
            The correction process:
            1. Applies five-point smoothing if enabled and n ≥ 5
            2. Applies cubic spline interpolation if enabled and n ≥ 3
            3. Ensures non-negative values in the result
        """
        result = np.asarray(modified_discharge).copy()

        # 步骤1：五点二次平滑
        if smoothing_enabled and self.n_points >= 5:
            result = self.five_point_smooth(result)

        # 步骤2：三次样条插值
        if interpolation_enabled and self.n_points >= 3:
            result = self.cubic_spline_interpolation(self.time_points, result)

        return result


def apply_smooth_correction(
    original_data: pd.DataFrame,
    modified_data: pd.DataFrame,
    discharge_column: str = "gen_discharge",
    time_column: str = "time",
) -> pd.DataFrame:
    """Apply smoothing correction algorithm based on the paper.

    This is the main interface function that supports multiple point modifications using
    five-point quadratic smoothing and cubic spline interpolation.

    Args:
        original_data (pd.DataFrame): Original data before any modifications.
        modified_data (pd.DataFrame): Data after user modifications.
        discharge_column (str, optional): Name of discharge column. Defaults to "gen_discharge".
        time_column (str, optional): Name of time column. Defaults to "time".

    Returns:
        pd.DataFrame: Data with smoothing correction applied.

    Raises:
        ValueError: If required columns are missing from the data.

    Note:
        The correction process:
        1. Validates input data
        2. Creates HydrographCorrector instance
        3. Applies smoothing and interpolation
        4. Returns corrected data
    """
    if discharge_column not in modified_data.columns:
        raise ValueError(f"数据中缺少径流列: {discharge_column}")

    if time_column not in modified_data.columns:
        raise ValueError(f"数据中缺少时间列: {time_column}")

    # 提取时间和径流数据
    time_points = (
        pd.to_datetime(modified_data[time_column]).astype(np.int64) / 1e9
    )  # 转换为秒数
    discharge_values = modified_data[discharge_column].values

    # 创建修正器
    corrector = HydrographCorrector(time_points, discharge_values)

    # 应用修正算法
    corrected_discharge = corrector.apply_correction(
        discharge_values, smoothing_enabled=True, interpolation_enabled=True
    )

    # 创建结果DataFrame
    result_data = modified_data.copy()
    result_data[discharge_column] = corrected_discharge

    return result_data


def apply_water_balance_correction(
    original_data: pd.DataFrame,
    modified_data: pd.DataFrame,
    discharge_column: str = "gen_discharge",
    time_column: str = "time",
    net_rain_column: str = "net_rain",
) -> pd.DataFrame:
    """Apply water balance correction to discharge data.

    Performs volume correction by calculating and applying a water balance coefficient
    to maintain consistency between net rainfall and discharge volumes.

    The correction process:
        1. Calculate total net rainfall volume
        2. Calculate total discharge volume
        3. Compute and apply volume correction factor

    Args:
        original_data (pd.DataFrame): Original data before any modifications.
        modified_data (pd.DataFrame): Data after user modifications.
        discharge_column (str, optional): Name of discharge column. Defaults to "gen_discharge".
        time_column (str, optional): Name of time column. Defaults to "time".
        net_rain_column (str, optional): Name of net rainfall column. Defaults to "net_rain".

    Returns:
        pd.DataFrame: Data with water balance correction applied.

    Raises:
        ValueError: If required columns are missing from the data.

    Note:
        - Assumes discharge and net_rain are in the same units (e.g., both in mm)
        - If net_rain_column is missing, returns the input data unchanged
        - Ensures all discharge values remain non-negative
    """
    # 1. 验证输入数据
    if discharge_column not in modified_data.columns:
        raise ValueError(f"数据中缺少径流列: {discharge_column}")
    if time_column not in modified_data.columns:
        raise ValueError(f"数据中缺少时间列: {time_column}")

    # 2. 提取径流数据
    discharge_values = modified_data[discharge_column].values

    # 3. 计算水量平衡
    if net_rain_column in modified_data.columns:
        # 计算净雨总量（假设单位与径流一致）
        total_net_rain = modified_data[net_rain_column].sum()

        # 计算径流总量（假设单位与净雨一致）
        total_discharge = np.sum(discharge_values)

        # 计算并应用水量平衡系数
        if total_net_rain > 0 and total_discharge > 0:
            # 计算修正系数
            volume_ratio = total_net_rain / total_discharge

            # 应用体积修正
            corrected_discharge = discharge_values * volume_ratio
        else:
            corrected_discharge = discharge_values
    else:
        corrected_discharge = discharge_values

    # 4. 确保所有值非负
    corrected_discharge = np.maximum(corrected_discharge, 0.0)

    # 5. 返回修正后的数据
    result_data = modified_data.copy()
    result_data[discharge_column] = corrected_discharge
    return result_data


def calculate_water_balance_metrics(
    data: pd.DataFrame,
    net_rain_column: str = "net_rain",
    discharge_column: str = "gen_discharge",
) -> dict:
    """Calculate water balance and discharge statistics metrics.

    Computes various metrics to assess water balance and discharge characteristics:
        - Total net rainfall
        - Total discharge volume
        - Water balance error (%)
        - Basic discharge statistics (mean, max, min, std)
        - Peak timing information

    Args:
        data (pd.DataFrame): Input data containing discharge and optionally net rainfall.
        net_rain_column (str, optional): Name of net rainfall column. Defaults to "net_rain".
        discharge_column (str, optional): Name of discharge column. Defaults to "gen_discharge".

    Returns:
        dict: Dictionary containing the following metrics:
            - total_net_rain (float): Total net rainfall (if available)
            - total_discharge (float): Total discharge volume
            - balance_error_percent (float): Water balance error percentage
            - discharge_stats (dict): Dictionary containing:
                - mean (float): Mean discharge
                - max (float): Maximum discharge
                - min (float): Minimum discharge
                - std (float): Standard deviation
                - peak_time_index: Index of peak discharge

    Note:
        - Assumes discharge and net_rain are in the same units
        - Water balance error is only calculated if net rainfall data is available
        - All statistics are computed ignoring any NaN values
    """
    metrics = {}

    # 计算净雨总量
    if net_rain_column in data.columns:
        total_net_rain = data[net_rain_column].sum()
        metrics["total_net_rain"] = total_net_rain

    # 计算径流总量
    if discharge_column in data.columns:
        discharge_volume = data[discharge_column].sum()
        metrics["total_discharge"] = discharge_volume

        # 计算水量平衡误差
        if net_rain_column in data.columns and total_net_rain > 0:
            balance_error = (discharge_volume - total_net_rain) / total_net_rain * 100
            metrics["balance_error_percent"] = balance_error

    # 径流统计
    if discharge_column in data.columns:
        discharge_stats = {
            "mean": data[discharge_column].mean(),
            "max": data[discharge_column].max(),
            "min": data[discharge_column].min(),
            "std": data[discharge_column].std(),
            "peak_time_index": data[discharge_column].idxmax(),
        }
        metrics["discharge_stats"] = discharge_stats

    return metrics


def validate_correction_quality(
    original_data: pd.DataFrame,
    corrected_data: pd.DataFrame,
    discharge_column: str = "gen_discharge",
) -> dict:
    """Validate the quality of hydrograph correction.

    Computes various metrics to assess how well the correction preserves important
    characteristics of the hydrograph while improving its quality.

    Args:
        original_data (pd.DataFrame): Original data before correction.
        corrected_data (pd.DataFrame): Data after correction.
        discharge_column (str, optional): Name of discharge column. Defaults to "gen_discharge".

    Returns:
        dict: Dictionary containing quality metrics:
            - mse (float): Mean squared error
            - rmse (float): Root mean squared error
            - mae (float): Mean absolute error
            - relative_error_percent (float): Mean relative error percentage
            - peak_preservation_ratio (float): Ratio of corrected to original peak
            - original_peak (float): Original peak discharge value
            - corrected_peak (float): Corrected peak discharge value

    Raises:
        ValueError: If discharge_column is missing from either dataset.

    Note:
        - All error metrics are computed between original and corrected values
        - Peak preservation ratio should ideally be close to 1.0
        - Relative error uses a small epsilon (1e-8) to avoid division by zero
    """
    if (
        discharge_column not in original_data.columns
        or discharge_column not in corrected_data.columns
    ):
        raise ValueError(f"数据中缺少径流列: {discharge_column}")

    original_values = original_data[discharge_column].values
    corrected_values = corrected_data[discharge_column].values

    # 计算质量指标
    mse = np.mean((corrected_values - original_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(corrected_values - original_values))

    # 相对误差
    relative_error = (
        np.mean(np.abs(corrected_values - original_values) / (original_values + 1e-8))
        * 100
    )

    # 峰值保持度
    original_peak = np.max(original_values)
    corrected_peak = np.max(corrected_values)
    peak_preservation = corrected_peak / original_peak if original_peak > 0 else 1.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "relative_error_percent": relative_error,
        "peak_preservation_ratio": peak_preservation,
        "original_peak": original_peak,
        "corrected_peak": corrected_peak,
    }
