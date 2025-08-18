"""
Author: Wenyu Ouyang
Date: 2025-08-18 11:46:35
LastEditTime: 2025-08-18 14:41:32
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
    """Test cases for streamflow_unit_conv function."""

    def test_streamflow_unit_conv_pint_quantities(self):
        """Test numpy/pandas input with pint quantities."""
        streamflow = np.array([1080.0, 2160.0]) * ureg.mm / ureg.h / 3
        area = np.array([1]) * ureg.km**2
        target_unit = "m^3/s"
        expected = np.array([100, 200])

        result = streamflow_unit_conv(streamflow, area, target_unit, True)
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "streamflow, area, source_unit, target_unit, area_unit, inverse, " "expected",
        [
            # Test case 1: m³/s to mm/3h conversion (numpy array, no area units)
            (
                np.array([100.0, 200.0, 300.0]),  # m³/s data
                np.array([1000]),  # km² (no units)
                "m^3/s",
                "mm/3h",
                "km^2",
                False,
                np.array([1.08, 2.16, 3.24]),
            ),
            # Test case 2: mm/3h to m³/s inverse conversion
            (
                np.array([1.08, 2.16, 3.24]),  # mm/3h data
                np.array([1000]),  # km² (no units)
                "mm/3h",
                "m^3/s",
                "km^2",
                True,
                np.array([100.0, 200.0, 300.0]),
            ),
            # Test case 3: m³/s to mm/6h conversion with different area unit
            (
                np.array([50.0, 100.0, 150.0]),  # m³/s data
                np.array([500000]),  # m² (no units)
                "m^3/s",
                "mm/6h",
                "m^2",
                False,
                np.array([2160.0, 4320.0, 6480.0]),
            ),
        ],
    )
    def test_streamflow_unit_conv_numpy_no_area_units(
        self, streamflow, area, source_unit, target_unit, area_unit, inverse, expected
    ):
        """Test conversion using source_unit and area_unit parameters."""
        result = streamflow_unit_conv(
            streamflow, area, target_unit, inverse, source_unit, area_unit
        )
        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_streamflow_unit_conv_identical_units(self):
        """Test that function returns original data when units are identical."""
        # Test case 1: xarray with identical units in attrs
        streamflow = xr.Dataset(
            {
                "streamflow": xr.DataArray(
                    np.array([[100, 200], [300, 400]]),
                    dims=["time", "basin"],
                    attrs={"units": "mm/d"},
                )
            }
        )
        area = xr.Dataset(
            {
                "area": xr.DataArray(
                    np.array([1, 2]), dims=["basin"], attrs={"units": "km^2"}
                )
            }
        )

        result = streamflow_unit_conv(streamflow, area, "mm/d", False)
        assert result is streamflow  # Should return exact same object

        # Test case 2: numpy array with explicit source_unit same as target
        streamflow_np = np.array([100.0, 200.0, 300.0])
        area_np = np.array([1000])
        result_np = streamflow_unit_conv(
            streamflow_np, area_np, "m^3/s", False, "m^3/s"
        )
        np.testing.assert_array_equal(result_np, streamflow_np)

        # Test case 3: pint quantity with identical units
        streamflow_pint = np.array([100.0, 200.0, 300.0]) * ureg.m**3 / ureg.s
        area_pint = np.array([1000]) * ureg.km**2
        result_pint = streamflow_unit_conv(streamflow_pint, area_pint, "m^3/s", False)
        assert result_pint is streamflow_pint

    def test_streamflow_unit_conv_inverse_validation(self):
        """Test validation of inverse parameter consistency."""
        streamflow = np.array([100.0, 200.0, 300.0])
        area = np.array([1000])

        # Valid conversions - should not raise
        result1 = streamflow_unit_conv(streamflow, area, "mm/d", False, "m^3/s")
        assert isinstance(result1, np.ndarray)

        result2 = streamflow_unit_conv(streamflow, area, "m^3/s", True, "mm/d")
        assert isinstance(result2, np.ndarray)

        # Invalid conversions - should raise ValueError
        with pytest.raises(ValueError, match="Inverse parameter.*inconsistent"):
            streamflow_unit_conv(streamflow, area, "mm/d", True, "m^3/s")

        with pytest.raises(ValueError, match="Inverse parameter.*inconsistent"):
            streamflow_unit_conv(streamflow, area, "m^3/s", False, "mm/d")


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
