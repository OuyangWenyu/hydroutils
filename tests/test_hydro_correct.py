"""
Test module for hydro_correct.py
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from hydroutils.hydro_correct import (
    apply_water_balance_correction,
    calculate_water_balance_metrics,
    HydrographCorrector
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # 创建一个10天的时间序列
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    
    # 创建一个简单的洪水过程线
    discharge = np.array([10, 15, 30, 50, 40, 25, 20, 15, 12, 10])
    
    # 创建原始数据
    original_data = pd.DataFrame({
        'time': dates,
        'gen_discharge': discharge.copy(),
        'net_rain': np.array([5, 8, 15, 20, 15, 10, 8, 5, 4, 3])
    })
    
    # 创建修改后的数据（修改第4个点的值）
    modified_data = original_data.copy()
    modified_data.loc[3, 'gen_discharge'] = 60  # 修改峰值
    
    return original_data, modified_data


def test_apply_water_balance_correction_basic(sample_data):
    """测试水量平衡修正的基本功能"""
    original_data, modified_data = sample_data
    
    # 应用修正
    result = apply_water_balance_correction(
        original_data, 
        modified_data,
        net_rain_column="net_rain",
        time_step_hours=24.0  # 日数据
    )
    
    # 基本检查
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(modified_data)
    assert 'gen_discharge' in result.columns
    assert 'time' in result.columns
    
    # 检查水量平衡
    total_net_rain = modified_data['net_rain'].sum()
    time_step_seconds = 24.0 * 3600  # 24小时转换为秒
    total_discharge = result['gen_discharge'].sum() * time_step_seconds / 1000  # 转换为mm
    
    # 允许1%的误差
    assert abs(total_discharge - total_net_rain) / total_net_rain < 0.01
    
    # 检查修正后的数据是否保持了用户修改的趋势
    # 修改点（第4个点）应该仍然是局部最大值
    peak_idx = 3
    assert result.iloc[peak_idx]['gen_discharge'] > result.iloc[peak_idx-1]['gen_discharge']
    assert result.iloc[peak_idx]['gen_discharge'] > result.iloc[peak_idx+1]['gen_discharge']
    
    # 检查所有值是否非负
    assert np.all(result['gen_discharge'] >= 0)


def test_apply_water_balance_correction_smoothness(sample_data):
    """测试修正后的过程是否平滑"""
    original_data, modified_data = sample_data
    
    result = apply_water_balance_correction(
        original_data, 
        modified_data,
        net_rain_column="net_rain",
        time_step_hours=24.0
    )
    
    # 计算一阶差分，检查平滑度
    discharge = result['gen_discharge'].values
    diff1 = np.diff(discharge)
    
    # 检查是否没有突变（差分不应该有太大的跳变）
    max_allowed_jump = 30  # 根据实际情况调整
    assert np.all(np.abs(diff1) < max_allowed_jump)
    
    # 检查修正后的过程是否比原始过程更平滑
    original_diff = np.abs(np.diff(modified_data['gen_discharge'].values))
    corrected_diff = np.abs(diff1)
    assert np.mean(corrected_diff) <= np.mean(original_diff)
    
    # 检查局部平滑性（三点移动平均的偏差）
    for i in range(1, len(discharge)-1):
        local_mean = (discharge[i-1] + discharge[i] + discharge[i+1]) / 3
        assert abs(discharge[i] - local_mean) < max_allowed_jump


def test_apply_water_balance_correction_no_changes(sample_data):
    """测试无修改点时的水量平衡修正"""
    original_data, _ = sample_data
    
    # 使用相同的数据作为原始和修改后的数据
    result = apply_water_balance_correction(
        original_data, 
        original_data,
        net_rain_column="net_rain",
        time_step_hours=24.0
    )
    
    # 检查结果的基本特性
    result_discharge = result['gen_discharge'].values
    original_discharge = original_data['gen_discharge'].values
    
    # 1. 检查水量平衡
    total_net_rain = original_data['net_rain'].sum()
    time_step_seconds = 24.0 * 3600
    total_discharge = np.sum(result_discharge) * time_step_seconds / 1000
    assert abs(total_discharge - total_net_rain) / total_net_rain < 0.01
    
    # 2. 检查总体趋势保持
    assert np.argmax(result_discharge) == np.argmax(original_discharge)
    assert np.corrcoef(result_discharge, original_discharge)[0, 1] > 0.9
    
    # 3. 检查数值范围合理
    assert np.min(result_discharge) >= 0  # 非负
    assert np.max(result_discharge) <= np.max(original_discharge) * 1.2  # 峰值不会显著增加
    
    # 4. 检查平滑性
    original_diff = np.abs(np.diff(original_discharge))
    result_diff = np.abs(np.diff(result_discharge))
    assert np.mean(result_diff) <= np.mean(original_diff)  # 平均变化率应该更小
    
    # 5. 检查体积修正后的一致性
    volume_ratio = total_net_rain / (np.sum(original_discharge) * time_step_seconds / 1000)
    expected_discharge = original_discharge * volume_ratio
    np.testing.assert_array_almost_equal(result_discharge, expected_discharge, decimal=2)


def test_apply_water_balance_correction_missing_columns():
    """测试缺少必要列时的错误处理"""
    # 创建测试数据
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    
    # 测试缺少discharge列
    bad_data = pd.DataFrame({
        'time': dates,
        'net_rain': [5, 8, 10, 8, 5]
    })
    with pytest.raises(ValueError, match="数据中缺少径流列"):
        apply_water_balance_correction(bad_data, bad_data)
    
    # 测试缺少time列
    bad_data = pd.DataFrame({
        'gen_discharge': [10, 20, 30, 40, 50],
        'net_rain': [5, 8, 10, 8, 5]
    })
    with pytest.raises(ValueError, match="数据中缺少时间列"):
        apply_water_balance_correction(bad_data, bad_data)
        
    # 测试缺少net_rain列时应该仍能工作
    data = pd.DataFrame({
        'time': dates,
        'gen_discharge': [10, 20, 30, 40, 50]
    })
    result = apply_water_balance_correction(data, data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)


def test_apply_water_balance_correction_short_series():
    """测试短时间序列的处理"""
    # 创建一个只有3个点的时间序列
    dates = pd.date_range(start='2023-01-01', periods=3, freq='D')
    discharge = np.array([10, 20, 15])
    net_rain = np.array([5, 10, 7])  # 总量22mm
    
    original_data = pd.DataFrame({
        'time': dates,
        'gen_discharge': discharge.copy(),
        'net_rain': net_rain
    })
    
    modified_data = original_data.copy()
    modified_data.loc[1, 'gen_discharge'] = 25  # 修改中间点
    
    # 应用修正
    result = apply_water_balance_correction(
        original_data, 
        modified_data,
        net_rain_column="net_rain",
        time_step_hours=24.0
    )
    
    # 1. 检查基本属性
    assert len(result) == 3
    assert result.iloc[1]['gen_discharge'] > result.iloc[0]['gen_discharge']
    assert result.iloc[1]['gen_discharge'] > result.iloc[2]['gen_discharge']
    
    # 2. 检查水量平衡
    total_net_rain = net_rain.sum()
    time_step_seconds = 24.0 * 3600
    total_discharge = result['gen_discharge'].sum() * time_step_seconds / 1000
    assert abs(total_discharge - total_net_rain) / total_net_rain < 0.01
    
    # 3. 检查用户修改点是否保持
    modified_idx = 1
    assert abs(result.iloc[modified_idx]['gen_discharge'] - 25) < 1.0  # 允许小的调整
    
    # 4. 检查平滑性
    discharge_values = result['gen_discharge'].values
    diff1 = np.diff(discharge_values)
    assert np.all(np.abs(diff1) < np.max(discharge_values) * 0.5)  # 变化不应过大


def test_apply_water_balance_correction_extreme_values():
    """测试极端值修改的处理"""
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    discharge = np.array([10, 15, 20, 15, 10])
    net_rain = np.array([5, 8, 10, 8, 5])  # 总量36mm
    
    original_data = pd.DataFrame({
        'time': dates,
        'gen_discharge': discharge.copy(),
        'net_rain': net_rain
    })
    
    modified_data = original_data.copy()
    modified_data.loc[2, 'gen_discharge'] = 1000  # 极端大的修改
    
    result = apply_water_balance_correction(
        original_data, 
        modified_data,
        net_rain_column="net_rain",
        time_step_hours=24.0
    )
    
    # 1. 检查水量平衡
    total_net_rain = net_rain.sum()
    time_step_seconds = 24.0 * 3600
    total_discharge = result['gen_discharge'].sum() * time_step_seconds / 1000
    assert abs(total_discharge - total_net_rain) / total_net_rain < 0.01
    
    # 2. 检查极端值是否被合理调整
    assert result.iloc[2]['gen_discharge'] > result.iloc[1]['gen_discharge']
    assert result.iloc[2]['gen_discharge'] > result.iloc[3]['gen_discharge']
    assert result.iloc[2]['gen_discharge'] < 1000  # 应该被调整到更合理的范围
    
    # 3. 检查基本约束
    assert np.all(result['gen_discharge'] >= 0)  # 非负
    assert np.all(np.diff(result['gen_discharge']) < 100)  # 变化不应过大
    
    # 4. 检查周围点的平滑过渡
    peak_idx = 2
    pre_diff = result.iloc[peak_idx]['gen_discharge'] - result.iloc[peak_idx-1]['gen_discharge']
    post_diff = result.iloc[peak_idx]['gen_discharge'] - result.iloc[peak_idx+1]['gen_discharge']
    assert abs(pre_diff - post_diff) < pre_diff * 0.5  # 前后变化应相对对称


def test_apply_water_balance_correction_metrics(sample_data):
    """测试修正结果的水量平衡指标"""
    original_data, modified_data = sample_data
    
    # 应用修正
    result = apply_water_balance_correction(
        original_data, 
        modified_data,
        net_rain_column="net_rain",
        time_step_hours=24.0
    )
    
    # 计算水量平衡指标
    metrics = calculate_water_balance_metrics(
        result,
        net_rain_column="net_rain",
        discharge_column="gen_discharge",
        time_step_hours=24.0
    )
    
    # 1. 检查基本指标存在性
    assert 'total_net_rain_mm' in metrics
    assert 'total_discharge_mm' in metrics
    assert 'balance_error_percent' in metrics
    assert 'discharge_stats' in metrics
    
    # 2. 检查水量平衡误差
    assert abs(metrics['balance_error_percent']) < 1.0  # 误差应小于1%
    
    # 3. 检查统计特征合理性
    stats = metrics['discharge_stats']
    assert stats['mean'] > 0
    assert stats['max'] >= stats['mean']
    assert stats['min'] <= stats['mean']
    assert stats['std'] >= 0
    
    # 4. 检查总量关系
    assert abs(metrics['total_discharge_mm'] - metrics['total_net_rain_mm']) < 1.0  # mm
    
    # 5. 检查峰值时间
    assert isinstance(stats['peak_time_index'], (int, np.integer))
    assert 0 <= stats['peak_time_index'] < len(result)
