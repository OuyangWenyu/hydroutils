"""
Author: Wenyu Ouyang
Date: 2025-08-18 11:46:35
LastEditTime: 2025-08-18 15:14:40
LastEditors: Wenyu Ouyang
Description: Unit tests for hydro_units module.
FilePath: \hydroutils\tests\test_hydro_units.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pint
import pytest
from datetime import datetime

from hydroutils.hydro_units import (
    streamflow_unit_conv,
    detect_time_interval,
    get_time_interval_info,
    validate_unit_compatibility,
    _normalize_unit,
    _is_inverse_conversion,
    _validate_inverse_consistency,
    _get_unit_conversion_info,
    _get_actual_source_unit,
)

# Create unit registry for pint
ureg = pint.UnitRegistry()
ureg.force_ndarray_like = True


class TestStreamflowUnitConv:
    """Test cases for streamflow_unit_conv function with new interface."""

    def test_numpy_array_conversion(self):
        """Test conversion with numpy arrays."""
        # Test m3/s to mm/d conversion
        flow_data = np.array([10.5, 15.2, 8.1, 12.7, 9.8])
        basin_area = np.array([1000.0])  # km2

        result = streamflow_unit_conv(
            flow_data, basin_area, "mm/d", source_unit="m^3/s"
        )
        expected = np.array([0.9072, 1.31328, 0.69984, 1.09728, 0.84672])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        # Test round-trip conversion
        result_back = streamflow_unit_conv(
            result, basin_area, "m^3/s", source_unit="mm/d"
        )
        np.testing.assert_array_almost_equal(result_back, flow_data, decimal=10)

    def test_pandas_series_conversion(self):
        """Test conversion with pandas Series."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        flow_data = pd.Series([2.1, 3.5, 1.8, 2.9, 2.3], index=dates)
        basin_area = pd.Series([500.0])

        result = streamflow_unit_conv(
            flow_data, basin_area, "m^3/s", source_unit="mm/h"
        )

        # Check that result preserves pandas structure
        assert isinstance(result, pd.Series)
        assert result.index.equals(dates)

        # Check values
        expected = np.array([291.666667, 486.111111, 250.0, 402.777778, 319.444444])
        np.testing.assert_array_almost_equal(result.values, expected, decimal=4)

    def test_xarray_dataset_conversion(self):
        """Test conversion with xarray Dataset."""
        time = pd.date_range("2024-01-01", periods=5, freq="h")
        flow_data = xr.Dataset(
            {"streamflow": (["time"], np.array([8.5, 12.1, 6.8, 15.2, 10.3]))},
            coords={"time": time},
        )
        flow_data["streamflow"].attrs["units"] = "m^3/s"

        area_data = xr.Dataset(
            {"area": (["basin"], np.array([750.0]))}, coords={"basin": ["basin_1"]}
        )
        area_data["area"].attrs["units"] = "km^2"

        result = streamflow_unit_conv(flow_data, area_data, "mm/h")

        # Check that result is xarray Dataset with correct units
        assert isinstance(result, xr.Dataset)
        assert result["streamflow"].attrs["units"] == "mm/h"

        # Check values approximately
        expected = np.array([0.0408, 0.05808, 0.03264, 0.07296, 0.04944])
        np.testing.assert_array_almost_equal(
            result["streamflow"].values, expected, decimal=4
        )

    def test_custom_units(self):
        """Test conversion with custom units like mm/3h."""
        flow_data = np.array([1.5, 2.2, 0.8, 3.1, 1.9])
        basin_area = np.array([1200.0])

        # Test mm/3h to m3/s
        result = streamflow_unit_conv(
            flow_data, basin_area, "m^3/s", source_unit="mm/3h"
        )
        expected = np.array(
            [166.66666667, 244.44444444, 88.88888889, 344.44444444, 211.11111111]
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

        # Test m3/s to mm/6h
        result2 = streamflow_unit_conv(result, basin_area, "mm/6h", source_unit="m^3/s")
        expected2 = np.array([3.0, 4.4, 1.6, 6.2, 3.8])
        np.testing.assert_array_almost_equal(result2, expected2, decimal=4)

    def test_automatic_unit_detection(self):
        """Test automatic unit detection from data attributes."""
        flow_data = xr.Dataset(
            {"flow": (["time"], np.array([5.2, 7.8, 4.1, 9.3, 6.7]))},
            coords={"time": pd.date_range("2024-01-01", periods=5)},
        )
        flow_data["flow"].attrs["units"] = "mm/d"

        basin_area = np.array([800.0])  # Uses default km2 unit

        result = streamflow_unit_conv(flow_data, basin_area, "m^3/s")
        expected = np.array(
            [48.14814815, 72.22222222, 37.96296296, 86.11111111, 62.03703704]
        )
        np.testing.assert_array_almost_equal(result["flow"].values, expected, decimal=4)

    def test_error_cases(self):
        """Test various error cases."""
        flow_data = np.array([1.0, 2.0, 3.0])
        basin_area = np.array([1000.0])

        # Test missing source unit for numpy array
        with pytest.raises(ValueError, match="No unit information found"):
            streamflow_unit_conv(flow_data, basin_area, "mm/d")

        # Test incompatible units
        with pytest.raises(ValueError, match="Incompatible units"):
            streamflow_unit_conv(flow_data, basin_area, "celsius", source_unit="m^3/s")

        # Test conflicting source unit
        flow_with_units = xr.Dataset({"flow": (["time"], flow_data, {"units": "mm/d"})})
        with pytest.raises(ValueError, match="conflicts with detected"):
            streamflow_unit_conv(
                flow_with_units, basin_area, "m^3/s", source_unit="m^3/s"
            )

    def test_same_units_optimization(self):
        """Test that function returns original data when units are identical."""
        flow_data = np.array([1.5, 2.1, 1.8])
        basin_area = np.array([500.0])

        result = streamflow_unit_conv(flow_data, basin_area, "mm/d", source_unit="mm/d")
        np.testing.assert_array_equal(result, flow_data)

    @pytest.mark.parametrize(
        "flow_values,area_values,source_unit,target_unit,area_unit,expected",
        [
            # m3/s to mm/d
            ([10.0, 20.0], [1000.0], "m^3/s", "mm/d", "km^2", [0.864, 1.728]),
            # mm/h to m3/s
            ([2.0, 4.0], [500.0], "mm/h", "m^3/s", "km^2", [277.777778, 555.555556]),
            # Custom units mm/3h to mm/h
            ([3.0, 6.0], [1000.0], "mm/3h", "mm/h", "km^2", [1.0, 2.0]),
        ],
    )
    def test_parametrized_conversions(
        self, flow_values, area_values, source_unit, target_unit, area_unit, expected
    ):
        """Test various unit conversions with parametrized inputs."""
        flow_data = np.array(flow_values)
        area_data = np.array(area_values)

        result = streamflow_unit_conv(
            flow_data,
            area_data,
            target_unit,
            source_unit=source_unit,
            area_unit=area_unit,
        )
        np.testing.assert_array_almost_equal(result, expected, decimal=4)


class TestDetectTimeInterval:
    """Test cases for detect_time_interval function."""

    def test_hourly_detection(self):
        """Test detection of hourly time intervals."""
        time_index = pd.date_range("2024-01-01", periods=5, freq="1h")
        result = detect_time_interval(time_index)
        assert result == "1h"

    def test_three_hourly_detection(self):
        """Test detection of 3-hourly time intervals."""
        time_index = pd.date_range("2024-01-01", periods=8, freq="3h")
        result = detect_time_interval(time_index)
        assert result == "3h"

    def test_daily_detection(self):
        """Test detection of daily time intervals."""
        time_index = pd.date_range("2024-01-01", periods=5, freq="1D")
        result = detect_time_interval(time_index)
        assert result == "1d"

    def test_list_input(self):
        """Test with list input."""
        time_list = ["2024-01-01 00:00", "2024-01-01 03:00", "2024-01-01 06:00"]
        result = detect_time_interval(time_list)
        assert result == "3h"

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        time_array = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T06:00:00", "2024-01-01T12:00:00"],
            dtype="datetime64",
        )
        result = detect_time_interval(time_array)
        assert result == "6h"

    def test_non_integer_hours_rounded(self):
        """Test that non-integer hours are rounded to nearest hour."""
        # Create times with 90-minute intervals (1.5 hours)
        base_time = datetime(2024, 1, 1, 0, 0)
        times = [base_time + pd.Timedelta(minutes=90 * i) for i in range(4)]
        result = detect_time_interval(times)
        assert result == "2h"  # Should round 1.5 to 2

    def test_insufficient_data_points(self):
        """Test error when insufficient data points."""
        time_index = pd.date_range("2024-01-01", periods=1, freq="1h")
        with pytest.raises(ValueError, match="at least 2 time points"):
            detect_time_interval(time_index)

    def test_multi_day_intervals(self):
        """Test detection of multi-day intervals."""
        time_index = pd.date_range("2024-01-01", periods=4, freq="3D")
        result = detect_time_interval(time_index)
        assert result == "3d"


class TestGetTimeIntervalInfo:
    """Test cases for get_time_interval_info function."""

    def test_valid_hour_intervals(self):
        """Test parsing valid hour interval strings."""
        assert get_time_interval_info("1h") == (1, "h")
        assert get_time_interval_info("3h") == (3, "h")
        assert get_time_interval_info("24h") == (24, "h")

    def test_valid_day_intervals(self):
        """Test parsing valid day interval strings."""
        assert get_time_interval_info("1d") == (1, "d")
        assert get_time_interval_info("7d") == (7, "d")
        assert get_time_interval_info("30d") == (30, "d")

    def test_invalid_formats(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid time interval format"):
            get_time_interval_info("3hours")

        with pytest.raises(ValueError, match="Invalid time interval format"):
            get_time_interval_info("h3")

        with pytest.raises(ValueError, match="Invalid time interval format"):
            get_time_interval_info("3")

        with pytest.raises(ValueError, match="Invalid time interval format"):
            get_time_interval_info("3m")  # minutes not supported


class TestValidateUnitCompatibility:
    """Test cases for validate_unit_compatibility function."""

    def test_compatible_depth_units(self):
        """Test that depth units are compatible with volume units."""
        assert validate_unit_compatibility("mm/3h", "m^3/s") is True
        assert validate_unit_compatibility("mm/d", "m^3/s") is True
        assert validate_unit_compatibility("mm/1h", "m^3/s") is True

    def test_compatible_volume_units(self):
        """Test that volume units are compatible with depth units."""
        assert validate_unit_compatibility("m^3/s", "mm/d") is True
        assert validate_unit_compatibility("m^3/s", "mm/3h") is True

    def test_compatible_same_category(self):
        """Test that units in same category are compatible."""
        assert validate_unit_compatibility("mm/h", "mm/d") is True
        assert validate_unit_compatibility("m^3/s", "m3/s") is True

    def test_incompatible_units(self):
        """Test that incompatible units return False."""
        assert validate_unit_compatibility("mm/h", "celsius") is False
        assert validate_unit_compatibility("m^3/s", "kg") is False
        assert validate_unit_compatibility("invalid", "mm/d") is False


class TestHelperFunctions:
    """Test cases for internal helper functions."""

    def test_normalize_unit(self):
        """Test unit normalization."""
        assert _normalize_unit("m3/s") == "m^3/s"
        assert _normalize_unit("ft3/s") == "ft^3/s"
        assert _normalize_unit("meter ** 3 / second") == "m^3/s"
        assert _normalize_unit("millimeter / day") == "mm/d"
        assert _normalize_unit("millimeter / hour") == "mm/h"

    def test_is_inverse_conversion(self):
        """Test inverse conversion detection."""
        assert _is_inverse_conversion("mm/d", "m^3/s") is True
        assert _is_inverse_conversion("m^3/s", "mm/d") is False
        assert _is_inverse_conversion("mm/3h", "m^3/s") is True
        assert _is_inverse_conversion("m^3/s", "mm/3h") is False

    def test_get_unit_conversion_info(self):
        """Test unit conversion info extraction."""
        assert _get_unit_conversion_info("mm/3h") == ("mm/h", 3)
        assert _get_unit_conversion_info("mm/5d") == ("mm/d", 5)
        assert _get_unit_conversion_info("mm/d") == ("mm/d", 1)
        assert _get_unit_conversion_info("mm/h") == ("mm/h", 1)

    def test_validate_inverse_consistency(self):
        """Test inverse parameter validation."""
        # Should not raise for correct combinations
        _validate_inverse_consistency("mm/d", "m^3/s", True)
        _validate_inverse_consistency("m^3/s", "mm/d", False)

        # Should raise for incorrect combinations
        with pytest.raises(ValueError, match="Inverse parameter.*inconsistent"):
            _validate_inverse_consistency("mm/d", "m^3/s", False)

        with pytest.raises(ValueError, match="Inverse parameter.*inconsistent"):
            _validate_inverse_consistency("m^3/s", "mm/d", True)

    def test_get_actual_source_unit(self):
        """Test source unit extraction from data."""
        # Test with explicit source_unit
        assert _get_actual_source_unit(np.array([1, 2, 3]), "mm/d") == "mm/d"

        # Test with pint quantity
        qty = np.array([1, 2, 3]) * ureg.m**3 / ureg.s
        result = _get_actual_source_unit(qty)
        # Normalize the result since pint might return different formats
        assert _normalize_unit(result) == "m^3/s"

        # Test with xarray with units in attrs
        ds = xr.Dataset({"data": xr.DataArray([1, 2, 3], attrs={"units": "mm/d"})})
        assert _get_actual_source_unit(ds) == "mm/d"

        # Test with numpy array without source_unit
        assert _get_actual_source_unit(np.array([1, 2, 3])) is None
