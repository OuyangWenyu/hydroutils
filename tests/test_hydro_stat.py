"""
Author: Wenyu Ouyang
Date: 2025-08-03
LastEditTime: 2025-08-04 09:12:43
LastEditors: Wenyu Ouyang
Description: Unit tests for hydro_stat module
FilePath: \hydroutils\tests\test_hydro_stat.py
Copyright (c) 2021-2025 MHPI group, Wenyu Ouyang. All rights reserved.
"""

import pytest
import numpy as np
from hydroutils.hydro_stat import (
    nse,
    rmse,
    mae,
    bias,
    pearson_r,
    r_squared,
    kge,
    mse,
    pbias,
    stat_error,
    stat_error_i,
    KGE,
    add_metric,
    HYDRO_METRICS,
    flood_peak_timing,
)


class TestDynamicMetricFunctions:
    """测试动态生成的指标函数"""

    @pytest.fixture
    def sample_data(self):
        """提供测试数据"""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        return observed, simulated

    @pytest.fixture
    def perfect_data(self):
        """提供完美匹配的测试数据"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        return data, data.copy()

    def test_nse_calculation(self, sample_data):
        """测试NSE计算"""
        observed, simulated = sample_data
        result = nse(observed, simulated)
        assert isinstance(result, (float, np.floating))
        assert -np.inf <= result <= 1.0

    def test_nse_perfect_match(self, perfect_data):
        """测试NSE完美匹配情况"""
        observed, simulated = perfect_data
        result = nse(observed, simulated)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_rmse_calculation(self, sample_data):
        """测试RMSE计算"""
        observed, simulated = sample_data
        result = rmse(observed, simulated)
        assert isinstance(result, (float, np.floating))
        assert result >= 0

    def test_rmse_perfect_match(self, perfect_data):
        """测试RMSE完美匹配情况"""
        observed, simulated = perfect_data
        result = rmse(observed, simulated)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_mae_calculation(self, sample_data):
        """测试MAE计算"""
        observed, simulated = sample_data
        result = mae(observed, simulated)
        assert isinstance(result, (float, np.floating))
        assert result >= 0

    def test_mae_perfect_match(self, perfect_data):
        """测试MAE完美匹配情况"""
        observed, simulated = perfect_data
        result = mae(observed, simulated)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_bias_calculation(self, sample_data):
        """测试Bias计算"""
        observed, simulated = sample_data
        result = bias(observed, simulated)
        assert isinstance(result, (float, np.floating))

    def test_bias_perfect_match(self, perfect_data):
        """测试Bias完美匹配情况"""
        observed, simulated = perfect_data
        result = bias(observed, simulated)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_pearson_r_calculation(self, sample_data):
        """测试Pearson相关系数计算"""
        observed, simulated = sample_data
        result = pearson_r(observed, simulated)
        assert isinstance(result, (float, np.floating))
        assert -1.0 <= result <= 1.0

    def test_pearson_r_perfect_match(self, perfect_data):
        """测试Pearson相关系数完美匹配情况"""
        observed, simulated = perfect_data
        result = pearson_r(observed, simulated)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_r_squared_calculation(self, sample_data):
        """测试R²计算"""
        observed, simulated = sample_data
        result = r_squared(observed, simulated)
        assert isinstance(result, (float, np.floating))
        assert result <= 1.0

    def test_r_squared_perfect_match(self, perfect_data):
        """测试R²完美匹配情况"""
        observed, simulated = perfect_data
        result = r_squared(observed, simulated)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_kge_calculation(self, sample_data):
        """测试KGE计算"""
        observed, simulated = sample_data
        result = kge(observed, simulated)
        assert isinstance(result, (float, np.floating))
        assert result <= 1.0

    def test_mse_calculation(self, sample_data):
        """测试MSE计算"""
        observed, simulated = sample_data
        result = mse(observed, simulated)
        assert isinstance(result, (float, np.floating))
        assert result >= 0

    def test_mse_perfect_match(self, perfect_data):
        """测试MSE完美匹配情况"""
        observed, simulated = perfect_data
        result = mse(observed, simulated)
        assert np.isclose(result, 0.0, atol=1e-10)

    def test_pbias_calculation(self, sample_data):
        """测试PBIAS计算"""
        observed, simulated = sample_data
        result = pbias(observed, simulated)
        assert isinstance(result, (float, np.floating))

    def test_pbias_perfect_match(self, perfect_data):
        """测试PBIAS完美匹配情况"""
        observed, simulated = perfect_data
        result = pbias(observed, simulated)
        assert np.isclose(result, 0.0, atol=1e-10)


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_arrays(self):
        """测试空数组"""
        result = nse(np.array([]), np.array([]))
        assert np.isnan(result)

    def test_single_value(self):
        """测试单个值"""
        result = nse(np.array([1.0]), np.array([1.1]))
        # Single value NSE can return -inf due to zero variance in denominator
        assert np.isinf(result) or np.isnan(result)

    def test_nan_values(self):
        """测试包含NaN值的数组"""
        observed = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        # 大多数指标应该能处理NaN值或返回NaN
        result = nse(observed, simulated)
        assert isinstance(result, (float, np.floating)) or np.isnan(result)

    def test_zero_variance_observed(self):
        """测试观测值方差为零的情况"""
        observed = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        simulated = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        # NSE在观测值方差为零时会返回一个数值（通常很小的负数）
        result = nse(observed, simulated)
        assert isinstance(result, (float, np.floating))


class TestExistingFunctions:
    """测试已存在的统计函数"""

    @pytest.fixture
    def sample_2d_data(self):
        """提供2D测试数据"""
        target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
        pred = np.array([[1.1, 2.2, 2.8, 4.1, 4.9], [2.1, 3.1, 3.9, 5.2, 5.8]])
        return target, pred

    def test_stat_error_i_calculation(self):
        """测试stat_error_i函数"""
        targ = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        result = stat_error_i(targ, pred)

        assert isinstance(result, dict)
        expected_keys = [
            "Bias",
            "RMSE",
            "ubRMSE",
            "Corr",
            "R2",
            "NSE",
            "KGE",
            "FHV",
            "FLV",
        ]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], (float, np.floating))

    def test_stat_error_calculation(self, sample_2d_data):
        """测试stat_error函数"""
        target, pred = sample_2d_data

        result = stat_error(target, pred)

        assert isinstance(result, dict)
        expected_keys = [
            "Bias",
            "RMSE",
            "ubRMSE",
            "Corr",
            "R2",
            "NSE",
            "KGE",
            "FHV",
            "FLV",
        ]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], np.ndarray)
            assert len(result[key]) == target.shape[0]

    def test_KGE_function(self):
        """测试原有的KGE函数"""
        xs = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        xo = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = KGE(xs, xo)

        assert isinstance(result, (float, np.floating))
        assert result <= 1.0


class TestDynamicFunctionGeneration:
    """测试动态函数生成系统"""

    def test_hydro_metrics_dict_exists(self):
        """测试HYDRO_METRICS字典存在"""
        assert HYDRO_METRICS is not None
        assert isinstance(HYDRO_METRICS, dict)
        assert len(HYDRO_METRICS) > 0

    def test_common_metrics_in_dict(self):
        """测试常见指标在字典中"""
        common_metrics = ["nse", "rmse", "mae", "bias", "pearson_r", "r_squared", "kge"]
        for metric in common_metrics:
            assert metric in HYDRO_METRICS

    def test_add_metric_function(self):
        """测试add_metric函数"""
        # 测试添加已存在的指标（应该成功）
        original_count = len(HYDRO_METRICS)
        add_metric("test_nse", "nse", "Test NSE function")

        # 验证函数已添加
        assert "test_nse" in HYDRO_METRICS
        assert len(HYDRO_METRICS) == original_count + 1

        # 测试新添加的函数是否可用
        from hydroutils.hydro_stat import test_nse

        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        result = test_nse(observed, simulated)
        assert isinstance(result, (float, np.floating))


class TestConsistency:
    """测试一致性"""

    def test_dynamic_vs_direct_nse(self):
        """测试动态生成的NSE与直接调用HydroErr的一致性"""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        # 使用动态生成的函数
        dynamic_result = nse(observed, simulated)

        # 直接调用HydroErr
        import HydroErr as he

        direct_result = he.nse(observed, simulated)

        assert np.isclose(dynamic_result, direct_result, rtol=1e-10)

    def test_all_common_metrics_work(self):
        """测试所有常见指标函数都能正常工作"""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        simulated = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        common_metrics = [
            "nse",
            "rmse",
            "mae",
            "bias",
            "pearson_r",
            "r_squared",
            "kge",
            "mse",
            "pbias",
        ]

        for metric_name in common_metrics:
            if metric_name in HYDRO_METRICS:
                # 动态导入函数
                from hydroutils import hydro_stat

                metric_func = getattr(hydro_stat, metric_name)

                # 测试函数调用
                result = metric_func(observed, simulated)
                assert isinstance(result, (float, np.floating)) or np.isnan(result)


class TestFloodPeakTiming:
    """测试mean_peak_timing函数"""

    @pytest.fixture
    def synthetic_peak_data(self):
        """生成有明显峰值的合成数据"""
        # 创建有两个峰值的观测数据
        t = np.arange(200)
        obs = 2 + np.sin(t * 0.1) + 0.5 * np.random.normal(0, 0.1, len(t))
        # 在位置50和150添加峰值
        obs[50] = 8.0
        obs[150] = 7.5

        # 模拟数据：峰值位置略有偏移
        sim = obs.copy()
        sim[52] = 7.8  # 峰值偏移2个时间步
        sim[50] = 3.0  # 原位置降低
        sim[153] = 7.2  # 峰值偏移3个时间步
        sim[150] = 3.5  # 原位置降低

        return obs, sim

    @pytest.fixture
    def no_peak_data(self):
        """生成没有明显峰值的平滑数据"""
        t = np.arange(100)
        obs = 2 + 0.1 * t + 0.05 * np.random.normal(0, 0.1, len(t))
        sim = obs + 0.1 * np.random.normal(0, 0.05, len(t))
        return obs, sim

    def test_mean_peak_timing_basic(self, synthetic_peak_data):
        """测试基本的峰值时间差计算"""
        obs, sim = synthetic_peak_data
        result = flood_peak_timing(obs, sim, window=5)

        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)
        assert result >= 0  # 时间差应该是非负数

    def test_mean_peak_timing_no_peaks(self, no_peak_data):
        """测试没有峰值的情况"""
        obs, sim = no_peak_data
        result = flood_peak_timing(obs, sim, window=5)

        # 如果没有检测到峰值，应该返回NaN
        assert np.isnan(result)

    def test_mean_peak_timing_perfect_match(self):
        """测试完美匹配的情况"""
        # 创建有峰值的数据
        obs = np.zeros(100)
        obs[25] = 10.0  # 峰值在位置25
        obs[75] = 8.0  # 峰值在位置75

        sim = obs.copy()  # 完美匹配

        result = flood_peak_timing(obs, sim, window=10)

        assert isinstance(result, (float, np.floating))
        assert np.isclose(result, 0.0, atol=0.1)  # 应该接近0

    def test_mean_peak_timing_with_nans(self):
        """测试包含NaN值的数据"""
        obs = np.ones(50) * 2.0
        obs[25] = 10.0  # 添加峰值
        obs[10] = np.nan  # 添加NaN

        sim = obs.copy()
        sim[27] = 9.5  # 峰值偏移2个位置
        sim[25] = 2.0  # 原位置降低
        sim[15] = np.nan  # 添加NaN

        result = flood_peak_timing(obs, sim, window=5)

        # 应该能处理NaN值并返回有效结果
        assert isinstance(result, (float, np.floating)) or np.isnan(result)

    def test_mean_peak_timing_empty_arrays(self):
        """测试空数组"""
        with pytest.raises(ValueError):
            flood_peak_timing(np.array([]), np.array([]))

    def test_mean_peak_timing_short_arrays(self):
        """测试太短的数组"""
        obs = np.array([1.0, 2.0])
        sim = np.array([1.1, 2.1])

        with pytest.raises(ValueError):
            flood_peak_timing(obs, sim)

    def test_mean_peak_timing_different_shapes(self):
        """测试不同形状的数组"""
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.1, 2.1])

        with pytest.raises(ValueError):
            flood_peak_timing(obs, sim)

    def test_mean_peak_timing_window_sizes(self, synthetic_peak_data):
        """测试不同窗口大小"""
        obs, sim = synthetic_peak_data

        # 测试不同窗口大小
        result_small = flood_peak_timing(obs, sim, window=2)
        result_large = flood_peak_timing(obs, sim, window=10)

        # 两种窗口大小都应该返回有效结果
        assert isinstance(result_small, (float, np.floating)) or np.isnan(result_small)
        assert isinstance(result_large, (float, np.floating)) or np.isnan(result_large)

    def test_mean_peak_timing_different_resolutions(self, synthetic_peak_data):
        """测试不同时间分辨率设置"""
        obs, sim = synthetic_peak_data

        # 测试不同分辨率
        result_hourly = flood_peak_timing(obs, sim, resolution="1H")
        result_daily = flood_peak_timing(obs, sim, resolution="1D")

        # 两种分辨率都应该返回有效结果
        assert isinstance(result_hourly, (float, np.floating)) or np.isnan(
            result_hourly
        )
        assert isinstance(result_daily, (float, np.floating)) or np.isnan(result_daily)

    def test_mean_peak_timing_constant_data(self):
        """测试常数数据（无变化）"""
        obs = np.ones(50) * 5.0
        sim = np.ones(50) * 5.0

        result = flood_peak_timing(obs, sim, window=5)

        # 常数数据可能没有峰值，应该返回NaN
        assert np.isnan(result)


if __name__ == "__main__":
    pytest.main([__file__])
