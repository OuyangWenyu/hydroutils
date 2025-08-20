"""
Author: MHPI group, Wenyu Ouyang
Date: 2021-12-31 11:08:29
LastEditTime: 2025-08-04 09:13:42
LastEditors: Wenyu Ouyang
Description: statistics calculation
FilePath: \hydroutils\hydroutils\hydro_stat.py
Copyright (c) 2021-2022 MHPI group, Wenyu Ouyang. All rights reserved.
"""

import copy
import itertools
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import HydroErr as he
import numpy as np
import scipy.stats
from scipy.stats import wilcoxon
from scipy import signal
import pandas as pd

ALL_METRICS = ["Bias", "RMSE", "ubRMSE", "Corr", "R2", "NSE", "KGE", "FHV", "FLV"]


def _validate_inputs(obs: np.ndarray, sim: np.ndarray) -> None:
    """Validate input arrays for peak timing analysis."""
    if not isinstance(obs, np.ndarray) or not isinstance(sim, np.ndarray):
        raise TypeError("Both obs and sim must be numpy arrays")

    if obs.shape != sim.shape:
        raise ValueError("obs and sim must have the same shape")

    if len(obs) < 3:
        raise ValueError(
            "Time series must have at least 3 data points for peak detection"
        )


def _mask_valid(obs: np.ndarray, sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove NaN values from both time series."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    return obs[mask], sim[mask]


def _get_frequency_factor(target_freq: str, current_freq: str) -> float:
    """Get frequency conversion factor between two pandas frequency strings."""
    # Simple conversion factors for common frequencies
    freq_to_hours = {
        "1H": 1,
        "3H": 3,
        "6H": 6,
        "12H": 12,
        "1D": 24,
        "D": 24,
    }

    target_hours = freq_to_hours.get(target_freq, 24)
    current_hours = freq_to_hours.get(current_freq, 24)

    return target_hours / current_hours


def fms(
    obs: np.ndarray, sim: np.ndarray, lower: float = 0.2, upper: float = 0.7
) -> float:
    r"""
    TODO: not fully tested
    Calculate the slope of the middle section of the flow duration curve [#]_

    .. math::
        \%\text{BiasFMS} = \frac{\left | \log(Q_{s,\text{lower}}) - \log(Q_{s,\text{upper}}) \right | -
            \left | \log(Q_{o,\text{lower}}) - \log(Q_{o,\text{upper}}) \right |}{\left |
            \log(Q_{s,\text{lower}}) - \log(Q_{s,\text{upper}}) \right |} \times 100,

    where :math:`Q_{s,\text{lower/upper}}` corresponds to the FDC of the simulations (here, `sim`) at the `lower` and
    `upper` bound of the middle section and :math:`Q_{o,\text{lower/upper}}` similarly for the observations (here,
    `obs`).

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    lower : float, optional
        Lower bound of the middle section in range ]0,1[, by default 0.2
    upper : float, optional
        Upper bound of the middle section in range ]0,1[, by default 0.7

    Returns
    -------
    float
        Slope of the middle section of the flow duration curve.

    References
    ----------
    .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process-based diagnostic approach to model
        evaluation: Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417,
        doi:10.1029/2007WR006716.
    """
    if len(obs) < 1:
        return np.nan

    if any((x <= 0) or (x >= 1) for x in [upper, lower]):
        raise ValueError("upper and lower have to be in range ]0,1[")

    if lower >= upper:
        raise ValueError("The lower threshold has to be smaller than the upper.")

    # get arrays of sorted (descending) discharges
    obs = np.sort(obs)
    sim = np.sort(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    # calculate fms part by part
    qsm_lower = np.log(sim[np.round(lower * len(sim)).astype(int)])
    qsm_upper = np.log(sim[np.round(upper * len(sim)).astype(int)])
    qom_lower = np.log(obs[np.round(lower * len(obs)).astype(int)])
    qom_upper = np.log(obs[np.round(upper * len(obs)).astype(int)])

    fms = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / (
        qom_lower - qom_upper + 1e-6
    )

    return fms * 100


def flood_volume_error(Q_obs, Q_sim, delta_t_seconds=10800):
    """
    Calculate relative flood volume error.

    Parameters
    ----------
    Q_obs : array-like
        Observed streamflow.
    Q_sim : array-like
        Simulated streamflow.
    delta_t_seconds : int, optional
        Time step in seconds, by default 10800 (3 hours).

    Returns
    -------
    float
        Relative flood volume error (%).
    """
    vol_obs = np.sum(Q_obs) * delta_t_seconds
    vol_sim = np.sum(Q_sim) * delta_t_seconds

    if vol_obs > 1e-6:
        return ((vol_sim - vol_obs) / vol_obs) * 100.0
    else:
        return np.nan


def flood_peak_error(Q_obs, Q_sim):
    """
    Calculate relative flood peak error.

    Parameters
    ----------
    Q_obs : array-like
        Observed streamflow.
    Q_sim : array-like
        Simulated streamflow.

    Returns
    -------
    float
        Relative flood peak error (%).
    """
    peak_obs = np.max(Q_obs)
    peak_sim = np.max(Q_sim)

    if peak_obs > 1e-6:
        return ((peak_sim - peak_obs) / peak_obs) * 100.0
    else:
        return np.nan


def flood_peak_timing(
    obs: np.ndarray,
    sim: np.ndarray,
    window: Optional[int] = None,
    resolution: str = "1D",
    datetime_coord: Optional[str] = None,
) -> float:
    """
    Calculate mean difference in peak flow timing (simplified version for numpy arrays).

    Uses scipy.find_peaks to find peaks in the observed time series. Starting with all observed peaks, those with a
    prominence of less than the standard deviation of the observed time series are discarded. Next, the lowest peaks
    are subsequently discarded until all remaining peaks have a distance of at least 100 steps. Finally, the
    corresponding peaks in the simulated time series are searched in a window of size `window` on either side of the
    observed peaks and the absolute time differences between observed and simulated peaks is calculated.
    The final metric is the mean absolute time difference across all peaks (in time steps).

    Parameters
    ----------
    obs : np.ndarray
        Observed time series.
    sim : np.ndarray
        Simulated time series.
    window : int, optional
        Size of window to consider on each side of the observed peak for finding the simulated peak. That is, the total
        window length to find the peak in the simulations is 2 * window + 1 centered at the observed
        peak. The default depends on the temporal resolution, e.g. for a resolution of '1D', a window of 3 is used and
        for a resolution of '1H' the window size is 12.
    resolution : str, optional
        Temporal resolution of the time series in pandas format, e.g. '1D' for daily and '1H' for hourly.
        Currently used only for determining default window size.
    datetime_coord : str, optional
        Name of datetime coordinate. Currently unused in this simplified implementation.

    Returns
    -------
    float
        Mean peak time difference in time steps. Returns NaN if no peaks are found.

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple
        meteorological datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci.,
        https://doi.org/10.5194/hess-2020-221
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations (scipy's find_peaks doesn't guarantee correctness with NaNs)
    obs_clean, sim_clean = _mask_valid(obs, sim)

    if len(obs_clean) < 3:
        return np.nan

    # determine default window size based on resolution
    if window is None:
        # infer a reasonable window size based on resolution
        window = max(int(_get_frequency_factor("12H", resolution)), 3)

    # heuristic to get indices of peaks and their corresponding height.
    # Use prominence based on standard deviation to filter significant peaks
    prominence_threshold = np.std(obs_clean)
    if prominence_threshold == 0:  # Handle constant time series
        prominence_threshold = (
            0.01 * np.mean(obs_clean) if np.mean(obs_clean) != 0 else 0.01
        )

    peaks, _ = signal.find_peaks(
        obs_clean, distance=100, prominence=prominence_threshold
    )

    if len(peaks) == 0:
        return np.nan

    # evaluate timing
    timing_errors = []
    for idx in peaks:
        # skip peaks at the start and end of the sequence
        if (idx - window < 0) or (idx + window >= len(obs_clean)):
            continue

        # find the corresponding peak in simulated data within the window
        window_start = max(0, idx - window)
        window_end = min(len(sim_clean), idx + window + 1)
        sim_window = sim_clean[window_start:window_end]

        # find the index of maximum value in the window
        local_peak_idx = np.argmax(sim_window)
        global_peak_idx = window_start + local_peak_idx

        # calculate the time difference between the peaks (in time steps)
        timing_error = abs(idx - global_peak_idx)
        timing_errors.append(timing_error)

    return np.mean(timing_errors) if timing_errors else np.nan


def KGE(xs: np.ndarray, xo: np.ndarray) -> float:
    """
    Kling Gupta Efficiency (Gupta et al., 2009, http://dx.doi.org/10.1016/j.jhydrol.2009.08.003)
    input:
        xs: simulated
        xo: observed
    output:
        KGE: Kling Gupta Efficiency
    """
    r = np.corrcoef(xo, xs)[0, 1]
    alpha = np.std(xs) / np.std(xo)
    beta = np.mean(xs) / np.mean(xo)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


# 定义指标函数映射字典，方便添加新指标
HYDRO_METRICS = {
    "nse": ("nse", "Nash-Sutcliffe Efficiency"),
    "rmse": ("rmse", "Root Mean Square Error"),
    "mae": ("mae", "Mean Absolute Error"),
    "bias": ("me", "Mean Error (Bias)"),
    "pearson_r": ("pearson_r", "Pearson correlation coefficient"),
    "r_squared": ("r_squared", "Coefficient of determination (R²)"),
    "kge": ("kge_2009", "Kling-Gupta Efficiency"),
    "mse": ("mse", "Mean Square Error"),
    "ve": ("ve", "Volumetric Efficiency"),
    "sa": ("sa", "Spectral Angle"),
    "sc": ("sc", "Spectral Correlation"),
    "sid": ("sid", "Spectral Information Divergence"),
    "sga": ("sga", "Spectral Gradient Angle"),
}


def _create_metric_function(
    he_func_name: str, description: str
) -> Callable[[np.ndarray, np.ndarray], float]:
    """工厂函数：动态创建指标计算函数"""

    def metric_func(
        observed: np.ndarray, simulated: np.ndarray, **kwargs: Any
    ) -> float:
        """
        {description}

        Parameters
        ----------
        observed : array-like
            Observed values
        simulated : array-like
            Simulated values
        **kwargs : dict
            Additional keyword arguments passed to the HydroErr function

        Returns
        -------
        float
            Calculated metric value
        """
        he_func = getattr(he, he_func_name)
        return he_func(observed, simulated, **kwargs)

    # 更新函数的文档字符串
    if metric_func.__doc__ is not None:
        metric_func.__doc__ = metric_func.__doc__.format(description=description)
    metric_func.__name__ = he_func_name

    return metric_func


# 动态生成所有指标函数并添加到当前模块
import sys

current_module = sys.modules[__name__]

for func_name, (he_func_name, description) in HYDRO_METRICS.items():
    # 检查HydroErr中是否存在该函数
    if hasattr(he, he_func_name):
        # 创建包装函数
        metric_func = _create_metric_function(he_func_name, description)
        # 将函数添加到当前模块
        setattr(current_module, func_name, metric_func)


def pbias(
    observed: Union[np.ndarray, List[float]], simulated: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculate Percent Bias (PBIAS)

    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values

    Returns
    -------
    float
        Percent bias value
    """
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    observed = observed[mask]
    simulated = simulated[mask]

    if len(observed) == 0:
        return np.nan

    return np.sum(simulated - observed) / np.sum(observed) * 100


def add_metric(func_name: str, he_func_name: str, description: str) -> None:
    """
    添加新的指标函数

    Parameters
    ----------
    func_name : str
        新函数的名称
    he_func_name : str
        HydroErr中对应函数的名称
    description : str
        函数描述
    """
    if hasattr(he, he_func_name):
        metric_func = _create_metric_function(he_func_name, description)
        setattr(current_module, func_name, metric_func)
        HYDRO_METRICS[func_name] = (he_func_name, description)
        print(f"已添加指标函数: {func_name}")
    else:
        print(f"警告: HydroErr中不存在函数 {he_func_name}")


def stat_error_i(targ_i: np.ndarray, pred_i: np.ndarray) -> Dict[str, float]:
    """Calculate multiple statistical metrics for one-dimensional arrays.

    This function computes a comprehensive set of statistical metrics comparing
    predicted values against target (observed) values. It handles NaN values
    and requires at least two valid data points for correlation-based metrics.

    Args:
        targ_i (np.ndarray): Target (observed) values.
        pred_i (np.ndarray): Predicted values.

    Returns:
        Dict[str, float]: Dictionary containing the following metrics:
            - Bias: Mean error
            - RMSE: Root mean square error
            - ubRMSE: Unbiased root mean square error
            - Corr: Pearson correlation coefficient
            - R2: Coefficient of determination
            - NSE: Nash-Sutcliffe efficiency
            - KGE: Kling-Gupta efficiency
            - FHV: Peak flow bias (top 2%)
            - FLV: Low flow bias (bottom 30%)

    Raises:
        ValueError: If there are fewer than 2 valid data points for correlation.

    Note:
        - NaN values are automatically handled (removed from calculations)
        - FHV and FLV are calculated in percentage
        - All metrics are calculated on valid (non-NaN) data points only

    Example:
        >>> target = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        >>> predicted = np.array([1.1, 2.2, 2.9, np.nan, 4.8])
        >>> metrics = stat_error_i(target, predicted)
        >>> print(metrics['RMSE'])  # Example output
        0.173
    """
    ind = np.where(np.logical_and(~np.isnan(pred_i), ~np.isnan(targ_i)))[0]
    # Theoretically at least two points for correlation
    if ind.shape[0] > 1:
        xx = pred_i[ind]
        yy = targ_i[ind]
        bias = he.me(xx, yy)
        # RMSE
        rmse = he.rmse(xx, yy)
        # ubRMSE
        pred_mean = np.nanmean(xx)
        target_mean = np.nanmean(yy)
        pred_anom = xx - pred_mean
        target_anom = yy - target_mean
        ubrmse = np.sqrt(np.nanmean((pred_anom - target_anom) ** 2))
        # rho R2 NSE
        corr = he.pearson_r(xx, yy)
        r2 = he.r_squared(xx, yy)
        nse = he.nse(xx, yy)
        kge = he.kge_2009(xx, yy)
        # percent bias
        pbias = np.sum(xx - yy) / np.sum(yy) * 100
        # FHV the peak flows bias 2%
        # FLV the low flows bias bottom 30%, log space
        pred_sort = np.sort(xx)
        target_sort = np.sort(yy)
        indexlow = round(0.3 * len(pred_sort))
        indexhigh = round(0.98 * len(pred_sort))
        lowpred = pred_sort[:indexlow]
        highpred = pred_sort[indexhigh:]
        lowtarget = target_sort[:indexlow]
        hightarget = target_sort[indexhigh:]
        pbiaslow = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
        pbiashigh = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
        return dict(
            Bias=bias,
            RMSE=rmse,
            ubRMSE=ubrmse,
            Corr=corr,
            R2=r2,
            NSE=nse,
            KGE=kge,
            FHV=pbiashigh,
            FLV=pbiaslow,
        )
    else:
        raise ValueError(
            "The number of data is less than 2, we don't calculate the statistics."
        )


def stat_error(
    target: np.ndarray, pred: np.ndarray, fill_nan: str = "no"
) -> Union[Dict[str, np.ndarray], Dict[str, List[float]]]:
    """Calculate statistical metrics for 2D arrays with NaN handling options.

    This function computes multiple statistical metrics comparing predicted values
    against target (observed) values for multiple time series (e.g., multiple
    basins). It provides different options for handling NaN values.

    Args:
        target (np.ndarray): Target (observed) values. 2D array [basin, sequence].
        pred (np.ndarray): Predicted values. Same shape as target.
        fill_nan (str, optional): Method for handling NaN values. Options:
            - "no": Ignore NaN values (default)
            - "sum": Sum values in NaN locations
            - "mean": Average values in NaN locations

    Returns:
        Union[Dict[str, np.ndarray], Dict[str, List[float]]]: Dictionary with metrics:
            - Bias: Mean error
            - RMSE: Root mean square error
            - ubRMSE: Unbiased root mean square error
            - Corr: Pearson correlation coefficient
            - R2: Coefficient of determination
            - NSE: Nash-Sutcliffe efficiency
            - KGE: Kling-Gupta efficiency
            - FHV: Peak flow bias (top 2%)
            - FLV: Low flow bias (bottom 30%)

    Raises:
        ValueError: If input arrays have wrong dimensions or incompatible shapes.

    Note:
        For fill_nan options:
        - "no": [1, nan, nan, 2] vs [0.3, 0.3, 0.3, 1.5] becomes [1, 2] vs [0.3, 1.5]
        - "sum": [1, nan, nan, 2] vs [0.3, 0.3, 0.3, 1.5] becomes [1, 2] vs [0.9, 1.5]
        - "mean": Similar to "sum" but takes average instead of sum

    Example:
        >>> target = np.array([[1.0, np.nan, np.nan, 2.0],
        ...                    [3.0, 4.0, np.nan, 6.0]])
        >>> pred = np.array([[1.1, 0.3, 0.3, 1.9],
        ...                  [3.2, 3.8, 0.5, 5.8]])
        >>> metrics = stat_error(target, pred, fill_nan="sum")
        >>> print(metrics['RMSE'])  # Example output
        array([0.158, 0.245])
    """
    if len(target.shape) == 3:
        raise ValueError(
            "The input data should be 2-dim, not 3-dim. If you want to calculate metrics for 3-d arrays, please use stat_errors function."
        )
    if type(fill_nan) is not str:
        raise ValueError("fill_nan should be a string.")
    if target.shape != pred.shape:
        raise ValueError("The shape of target and pred should be the same.")
    if fill_nan != "no":
        each_non_nan_idx = []
        all_non_nan_idx: list[int] = []
        for i in range(target.shape[0]):
            tmp = target[i]
            non_nan_idx_tmp = [j for j in range(tmp.size) if not np.isnan(tmp[j])]
            each_non_nan_idx.append(non_nan_idx_tmp)
            # TODO: now all_non_nan_idx is only set for ET, because of its irregular nan values
            all_non_nan_idx = all_non_nan_idx + non_nan_idx_tmp
            non_nan_idx = np.unique(all_non_nan_idx).tolist()
        # some NaN data appear in different dates in different basins, so we have to calculate the metric for each basin
        # but for ET, it is not very resonable to calculate the metric for each basin in this way, for example,
        # the non_nan_idx: [1, 9, 17, 33, 41], then there are 16 elements in 17 -> 33, so use all_non_nan_idx is better
        # hence we don't use each_non_nan_idx finally
        out_dict: Dict[str, List[float]] = dict(
            Bias=[],
            RMSE=[],
            ubRMSE=[],
            Corr=[],
            R2=[],
            NSE=[],
            KGE=[],
            FHV=[],
            FLV=[],
        )
    if fill_nan == "sum":
        for i in range(target.shape[0]):
            tmp = target[i]
            # non_nan_idx = each_non_nan_idx[i]
            targ_i = tmp[non_nan_idx]
            pred_i = np.add.reduceat(pred[i], non_nan_idx)
            dict_i = stat_error_i(targ_i, pred_i)
            out_dict["Bias"].append(dict_i["Bias"])
            out_dict["RMSE"].append(dict_i["RMSE"])
            out_dict["ubRMSE"].append(dict_i["ubRMSE"])
            out_dict["Corr"].append(dict_i["Corr"])
            out_dict["R2"].append(dict_i["R2"])
            out_dict["NSE"].append(dict_i["NSE"])
            out_dict["KGE"].append(dict_i["KGE"])
            out_dict["FHV"].append(dict_i["FHV"])
            out_dict["FLV"].append(dict_i["FLV"])
        return out_dict
    elif fill_nan == "mean":
        for i in range(target.shape[0]):
            tmp = target[i]
            # non_nan_idx = each_non_nan_idx[i]
            targ_i = tmp[non_nan_idx]
            pred_i_sum = np.add.reduceat(pred[i], non_nan_idx)
            if non_nan_idx[-1] < len(pred[i]):
                idx4mean = non_nan_idx + [len(pred[i])]
            else:
                idx4mean = copy.copy(non_nan_idx)
            idx_interval = [y - x for x, y in zip(idx4mean, idx4mean[1:])]
            pred_i = pred_i_sum / idx_interval
            dict_i = stat_error_i(targ_i, pred_i)
            out_dict["Bias"].append(dict_i["Bias"])
            out_dict["RMSE"].append(dict_i["RMSE"])
            out_dict["ubRMSE"].append(dict_i["ubRMSE"])
            out_dict["Corr"].append(dict_i["Corr"])
            out_dict["R2"].append(dict_i["R2"])
            out_dict["NSE"].append(dict_i["NSE"])
            out_dict["KGE"].append(dict_i["KGE"])
            out_dict["FHV"].append(dict_i["FHV"])
            out_dict["FLV"].append(dict_i["FLV"])
        return out_dict
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target) ** 2, axis=1))
    # ubRMSE
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom) ** 2, axis=1))
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    KGe = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    num_lowtarget_zero = 0
    for k in range(ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100
            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                yymean = yy.mean()
                SST: float = np.sum((yy - yymean) ** 2)
                SSReg: float = np.sum((xx - yymean) ** 2)
                SSRes: float = np.sum((yy - xx) ** 2)
                R2[k] = 1 - SSRes / SST
                NSE[k] = 1 - SSRes / SST
                KGe[k] = KGE(xx, yy)
            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.98 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            if np.sum(lowtarget) == 0:
                num_lowtarget_zero = num_lowtarget_zero + 1
            with warnings.catch_warnings():
                # Sometimes the lowtarget is all 0, which will cause a warning
                # but I know it is not an error, so I ignore it
                warnings.simplefilter("ignore", category=RuntimeWarning)
                PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
            PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
    outDict = dict(
        Bias=Bias,
        RMSE=RMSE,
        ubRMSE=ubRMSE,
        Corr=Corr,
        R2=R2,
        NSE=NSE,
        KGE=KGe,
        FHV=PBiashigh,
        FLV=PBiaslow,
    )
    # "The CDF of BFLV will not reach 1.0 because some basins have all zero flow observations for the "
    # "30% low flow interval, the percent bias can be infinite\n"
    # "The number of these cases is " + str(num_lowtarget_zero)
    return outDict


def stat_errors(
    target: np.ndarray, pred: np.ndarray, fill_nan: Optional[List[str]] = None
) -> List[Dict[str, np.ndarray]]:
    """Calculate statistical metrics for 3D arrays with multiple variables.

    This function extends stat_error to handle 3D arrays where the third dimension
    represents different variables. Each variable can have its own NaN handling
    method.

    Args:
        target (np.ndarray): Target (observed) values. 3D array [basin, sequence, variable].
        pred (np.ndarray): Predicted values. Same shape as target.
        fill_nan (List[str], optional): List of NaN handling methods, one per variable.
            Each element can be "no", "sum", or "mean". Defaults to ["no"].

    Returns:
        List[Dict[str, np.ndarray]]: List of dictionaries, one per variable.
            Each dictionary contains:
            - Bias: Mean error
            - RMSE: Root mean square error
            - ubRMSE: Unbiased root mean square error
            - Corr: Pearson correlation coefficient
            - R2: Coefficient of determination
            - NSE: Nash-Sutcliffe efficiency
            - KGE: Kling-Gupta efficiency
            - FHV: Peak flow bias (top 2%)
            - FLV: Low flow bias (bottom 30%)

    Raises:
        ValueError: If:
            - Input arrays are not 3D
            - Arrays have incompatible shapes
            - fill_nan length doesn't match number of variables

    Example:
        >>> target = np.array([[[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]]])  # 1x3x2
        >>> pred = np.array([[[1.1, 2.1], [3.0, 3.9], [4.9, 5.8]]])
        >>> metrics = stat_errors(target, pred, fill_nan=["no", "sum"])
        >>> print(len(metrics))  # Number of variables
        2
        >>> print(metrics[0]['RMSE'])  # RMSE for first variable
        array([0.141])
    """
    if fill_nan is None:
        fill_nan = ["no"]
    if len(target.shape) != 3:
        raise ValueError(
            "The input data should be 3-dim, not 2-dim. If you want to calculate "
            "metrics for 2-d arrays, please use stat_error function."
        )
    if target.shape != pred.shape:
        raise ValueError("The shape of target and pred should be the same.")
    if type(fill_nan) is not list or len(fill_nan) != target.shape[-1]:
        raise ValueError(
            "Please give same length of fill_nan as the number of variables."
        )
    dict_list = []
    for k in range(target.shape[-1]):
        k_dict = stat_error(target[:, :, k], pred[:, :, k], fill_nan=fill_nan[k])
        dict_list.append(k_dict)
    return dict_list


def cal_4_stat_inds(b: np.ndarray) -> List[float]:
    """Calculate four basic statistical indices for an array.

    This function computes four common statistical measures: 10th and 90th
    percentiles, mean, and standard deviation. If the standard deviation is
    very small (< 0.001), it is set to 1 to avoid numerical issues.

    Args:
        b (np.ndarray): Input array of numerical values.

    Returns:
        List[float]: Four statistical measures in order:
            - p10: 10th percentile
            - p90: 90th percentile
            - mean: Arithmetic mean
            - std: Standard deviation (minimum 0.001)

    Note:
        - NaN values should be removed before calling this function
        - If std < 0.001, it is set to 1 to avoid division issues
        - All returned values are cast to float type

    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> p10, p90, mean, std = cal_4_stat_inds(data)
        >>> print(f"P10: {p10}, P90: {p90}, Mean: {mean}, Std: {std}")
        P10: 1.9, P90: 9.1, Mean: 5.5, Std: 2.87
    """
    p10: float = np.percentile(b, 10).astype(float)
    p90: float = np.percentile(b, 90).astype(float)
    mean: float = np.mean(b).astype(float)
    std: float = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def cal_stat(x: np.ndarray) -> List[float]:
    """Calculate basic statistics for an array, handling NaN values.

    This function computes four basic statistical measures (10th and 90th
    percentiles, mean, and standard deviation) while properly handling NaN
    values. If the array is empty after removing NaN values, a zero value
    is used for calculations.

    Args:
        x (np.ndarray): Input array, may contain NaN values.

    Returns:
        List[float]: Four statistical measures in order:
            - p10: 10th percentile
            - p90: 90th percentile
            - mean: Arithmetic mean
            - std: Standard deviation (minimum 0.001)

    Note:
        - NaN values are automatically removed before calculations
        - If all values are NaN, returns statistics for [0]
        - Uses cal_4_stat_inds for actual calculations
        - If std < 0.001, it is set to 1 to avoid division issues

    Example:
        >>> data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        >>> p10, p90, mean, std = cal_stat(data)
        >>> print(f"P10: {p10}, P90: {p90}, Mean: {mean}, Std: {std}")
        P10: 1.3, P90: 4.7, Mean: 3.0, Std: 1.58
    """
    a = x.flatten()
    b = a[~np.isnan(a)]
    if b.size == 0:
        # if b is [], then give it a 0 value
        b = np.array([0])
    return cal_4_stat_inds(b)


def cal_stat_gamma(x: np.ndarray) -> List[float]:
    """Transform time series data to approximate normal distribution.

    This function applies a transformation to hydrological time series data
    (streamflow, precipitation, evapotranspiration) to make it more normally
    distributed. The transformation is: log10(sqrt(x) + 0.1).

    Args:
        x (np.ndarray): Time series data, typically daily values of:
            - Streamflow
            - Precipitation
            - Evapotranspiration

    Returns:
        List[float]: Four statistical measures of transformed data:
            - p10: 10th percentile
            - p90: 90th percentile
            - mean: Arithmetic mean
            - std: Standard deviation (minimum 0.001)

    Note:
        - NaN values are automatically removed before transformation
        - Transformation: log10(sqrt(x) + 0.1)
        - This transformation helps handle gamma-distributed data
        - If std < 0.001, it is set to 1 to avoid division issues

    Example:
        >>> data = np.array([0.0, 0.1, 1.0, 10.0, np.nan, 100.0])
        >>> p10, p90, mean, std = cal_stat_gamma(data)
        >>> print(f"P10: {p10:.2f}, P90: {p90:.2f}")
        P10: -0.52, P90: 1.01
    """
    a = x.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    return cal_4_stat_inds(b)


def cal_stat_prcp_norm(x, meanprep):
    """Normalize variables by precipitation and calculate gamma statistics.

    This function normalizes a variable (e.g., streamflow) by mean precipitation
    to remove the influence of rainfall magnitude, making statistics comparable
    between dry and wet basins. After normalization, gamma transformation is
    applied.

    Args:
        x (np.ndarray): Data to be normalized, typically streamflow or other
            hydrological variables.
        meanprep (np.ndarray): Mean precipitation values for normalization.
            Usually obtained from basin attributes (e.g., p_mean).

    Returns:
        List[float]: Four statistical measures of normalized data:
            - p10: 10th percentile
            - p90: 90th percentile
            - mean: Arithmetic mean
            - std: Standard deviation (minimum 0.001)

    Note:
        - Normalization: x / meanprep (unit: mm/day / mm/day)
        - After normalization, gamma transformation is applied
        - Helps compare basins with different precipitation regimes
        - If std < 0.001, it is set to 1 to avoid division issues

    Example:
        >>> data = np.array([[10.0, 20.0], [30.0, 40.0]])  # 2 basins, 2 timesteps
        >>> mean_prep = np.array([100.0, 200.0])  # Mean prep for 2 basins
        >>> p10, p90, mean, std = cal_stat_prcp_norm(data, mean_prep)
        >>> print(f"P10: {p10:.3f}, P90: {p90:.3f}")
        P10: -0.523, P90: -0.398
    """
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    # unit (mm/day)/(mm/day)
    flowua = x / tempprep
    return cal_stat_gamma(flowua)


def trans_norm(x, var_lst, stat_dict, *, to_norm):
    """Normalize or denormalize data using statistical parameters.

    This function performs normalization or denormalization on 2D or 3D data
    arrays using pre-computed statistical parameters. It supports multiple
    variables and can handle both site-based and time series data.

    Args:
        x (np.ndarray): Input data array:
            - 2D: [sites, variables]
            - 3D: [sites, time, variables]
        var_lst (Union[str, List[str]]): Variable name(s) to process.
        stat_dict (Dict[str, List[float]]): Dictionary containing statistics
            for each variable. Each value is [p10, p90, mean, std].
        to_norm (bool): If True, normalize data; if False, denormalize data.

    Returns:
        np.ndarray: Normalized or denormalized data with same shape as input.

    Note:
        - Normalization: (x - mean) / std
        - Denormalization: x * std + mean
        - Statistics should be pre-computed for each variable
        - Handles single variable (str) or multiple variables (list)
        - Preserves input array dimensions

    Example:
        >>> # Normalization example
        >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 sites, 2 variables
        >>> stats = {'var1': [0, 2, 1, 0.5], 'var2': [1, 5, 3, 1.0]}
        >>> vars = ['var1', 'var2']
        >>> normalized = trans_norm(data, vars, stats, to_norm=True)
        >>> print(normalized)  # Example output
        array([[0. , -1.],
               [4. ,  1.]])
    """
    if type(var_lst) is str:
        var_lst = [var_lst]
    out = np.zeros(x.shape)
    for k in range(len(var_lst)):
        var = var_lst[k]
        stat = stat_dict[var]
        if to_norm is True:
            if len(x.shape) == 3:
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        elif len(x.shape) == 3:
            out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
        elif len(x.shape) == 2:
            out[:, k] = x[:, k] * stat[3] + stat[2]
    return out


def ecdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Empirical Cumulative Distribution Function (ECDF).

    This function calculates the empirical CDF for a given dataset. The ECDF
    shows the fraction of observations less than or equal to each data point.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays:
            - x: Sorted input data
            - y: Cumulative probabilities (0 to 1)

    Note:
        - Data is sorted in ascending order
        - Probabilities are calculated as (i)/(n) for i=1..n
        - No special handling of NaN values - remove them before calling

    Example:
        >>> data = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
        >>> x, y = ecdf(data)
        >>> print("Values:", x)
        Values: [1 2 2 3 3 3 4 4 5]
        >>> print("Probabilities:", y)
        Probabilities: [0.111 0.222 0.333 0.444 0.556 0.667 0.778 0.889 1.000]
    """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return (x, y)


def wilcoxon_t_test(xs: np.ndarray, xo: np.ndarray) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test on paired samples.

    This function performs a Wilcoxon signed-rank test to determine whether two
    related samples have the same distribution. It's particularly useful for
    comparing model predictions against observations.

    Args:
        xs (np.ndarray): First sample (typically simulated/predicted values).
        xo (np.ndarray): Second sample (typically observed values).

    Returns:
        Tuple[float, float]: Test statistics:
            - w: Wilcoxon test statistic
            - p: p-value for the test

    Note:
        - Non-parametric alternative to paired t-test
        - Assumes samples are paired and same length
        - Direction of difference (xs-xo vs xo-xs) doesn't affect results
        - Uses scipy.stats.wilcoxon under the hood

    Example:
        >>> sim = np.array([102, 104, 98, 101, 96, 103, 95])
        >>> obs = np.array([100, 102, 95, 100, 93, 101, 94])
        >>> w, p = wilcoxon_t_test(sim, obs)
        >>> print(f"W-statistic: {w:.2f}, p-value: {p:.4f}")
        W-statistic: 26.50, p-value: 0.0234
    """
    diff = xs - xo  # same result when using xo-xs
    w, p = wilcoxon(diff)
    return w, p


def wilcoxon_t_test_for_lst(x_lst, rnd_num=2):
    """Perform pairwise Wilcoxon tests on multiple arrays.

    This function performs Wilcoxon signed-rank tests on every possible pair
    of arrays in a list of arrays. Results are rounded to specified precision.

    Args:
        x_lst (List[np.ndarray]): List of arrays to compare pairwise.
        rnd_num (int, optional): Number of decimal places to round results to.
            Defaults to 2.

    Returns:
        Tuple[List[float], List[float]]: Two lists:
            - w: List of Wilcoxon test statistics for each pair
            - p: List of p-values for each pair

    Note:
        - Generates all possible pairs using itertools.combinations
        - Results are ordered by pair combinations
        - Number of pairs = n*(n-1)/2 where n is number of arrays
        - All test statistics and p-values are rounded

    Example:
        >>> arrays = [
        ...     np.array([1, 2, 3, 4]),
        ...     np.array([2, 3, 4, 5]),
        ...     np.array([3, 4, 5, 6])
        ... ]
        >>> w, p = wilcoxon_t_test_for_lst(arrays)
        >>> print(f"W-statistics: {w}")
        W-statistics: [0.00, 0.00, 0.00]
        >>> print(f"p-values: {p}")
        p-values: [0.07, 0.07, 0.07]
    """
    arr_lst = np.asarray(x_lst)
    w, p = [], []
    arr_lst_pair = list(itertools.combinations(arr_lst, 2))
    for arr_pair in arr_lst_pair:
        wi, pi = wilcoxon_t_test(arr_pair[0], arr_pair[1])
        w.append(round(wi, rnd_num))
        p.append(round(pi, rnd_num))
    return w, p


def cal_fdc(data: np.array, quantile_num=100):
    """Calculate Flow Duration Curves (FDC) for multiple time series.

    This function computes flow duration curves for multiple time series data,
    typically used for analyzing streamflow characteristics. It handles NaN
    values and provides a specified number of quantile points.

    Args:
        data (np.array): 2D array of shape [n_grid, n_day] containing time
            series data for multiple locations/grids.
        quantile_num (int, optional): Number of quantile points to compute
            for each FDC. Defaults to 100.

    Returns:
        np.ndarray: Array of shape [n_grid, quantile_num] containing FDC
            values for each location/grid.

    Note:
        - Data is sorted from high to low flow
        - NaN values are removed before processing
        - Empty series are filled with zeros
        - Quantiles are evenly spaced from 0 to 1
        - Output shape is always [n_grid, quantile_num]

    Raises:
        Exception: If output flow array length doesn't match quantile_num.

    Example:
        >>> data = np.array([
        ...     [10, 8, 6, 4, 2],  # First location
        ...     [20, 16, 12, 8, 4]  # Second location
        ... ])
        >>> fdc = cal_fdc(data, quantile_num=5)
        >>> print(fdc)
        array([[10.,  8.,  6.,  4.,  2.],
               [20., 16., 12.,  8.,  4.]])
    """
    # data = n_grid * n_day
    n_grid, n_day = data.shape
    fdc = np.full([n_grid, quantile_num], np.nan)
    for ii in range(n_grid):
        temp_data0 = data[ii, :]
        temp_data = temp_data0[~np.isnan(temp_data0)]
        # deal with no data case for some gages
        if len(temp_data) == 0:
            temp_data = np.full(n_day, 0)
        # sort from large to small
        temp_sort = np.sort(temp_data)[::-1]
        # select quantile_num quantile points
        n_len = len(temp_data)
        ind = (np.arange(quantile_num) / quantile_num * n_len).astype(int)
        fdc_flow = temp_sort[ind]
        if len(fdc_flow) != quantile_num:
            raise Exception("unknown assimilation variable")
        else:
            fdc[ii, :] = fdc_flow

    return fdc


def remove_abnormal_data(data, *, q1=0.00001, q2=0.99999):
    """Remove extreme values from data using quantile thresholds.

    This function removes data points that fall outside specified quantile
    ranges by replacing them with NaN values. This is useful for removing
    outliers or extreme values that might affect analysis.

    Args:
        data (np.ndarray): Input data array.
        q1 (float, optional): Lower quantile threshold. Values below this
            quantile will be replaced with NaN. Defaults to 0.00001.
        q2 (float, optional): Upper quantile threshold. Values above this
            quantile will be replaced with NaN. Defaults to 0.99999.

    Returns:
        np.ndarray: Data array with extreme values replaced by NaN.

    Note:
        - Uses numpy.quantile for threshold calculation
        - Values equal to thresholds are kept
        - Original array shape is preserved
        - NaN values in input are preserved
        - Default thresholds keep 99.998% of data

    Example:
        >>> data = np.array([1, 2, 3, 100, 4, 5, 0.001, 6])
        >>> cleaned = remove_abnormal_data(data, q1=0.1, q2=0.9)
        >>> print(cleaned)
        array([nan,  2.,  3.,  nan,  4.,  5.,  nan,  6.])
    """
    # remove abnormal data
    data[data < np.quantile(data, q1)] = np.nan
    data[data > np.quantile(data, q2)] = np.nan
    return data


def month_stat_for_daily_df(df):
    """Calculate monthly statistics from daily data.

    This function resamples daily data to monthly frequency by computing the
    mean value for each month. It ensures the input DataFrame has a datetime
    index before resampling.

    Args:
        df (pd.DataFrame): DataFrame containing daily data with datetime index
            or index that can be converted to datetime.

    Returns:
        pd.DataFrame: DataFrame containing monthly statistics (means).
            Index is the start of each month.

    Note:
        - Uses pandas resample with 'MS' (month start) frequency
        - Automatically converts index to datetime if needed
        - Computes mean value for each month
        - Handles missing values according to pandas defaults

    Example:
        >>> dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        >>> data = pd.DataFrame({'value': range(366)}, index=dates)
        >>> monthly = month_stat_for_daily_df(data)
        >>> print(monthly.head())
                       value
        2020-01-01  15.0
        2020-02-01  45.5
        2020-03-01  74.0
        2020-04-01  105.0
        2020-05-01  135.5
    """
    # guarantee the index is datetime
    df.index = pd.to_datetime(df.index)
    return df.resample("MS").mean()
