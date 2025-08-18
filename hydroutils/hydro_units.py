"""
Hydrological unit conversion utilities.

This module provides comprehensive unit conversion functionality for hydrological data,
including streamflow unit conversions between depth units (mm/time) and volume units (m³/s).
"""

import re
import numpy as np
import pandas as pd
import pint
import xarray as xr
from datetime import datetime, timedelta
from typing import Union, Tuple, Optional

# Create unit registry for pint
ureg = pint.UnitRegistry()


def _convert_target_unit(target_unit):
    """Convert user-friendly unit to standard unit for internal calculations."""
    if match := re.match(r"mm/(\d+)(h|d)", target_unit):
        num, unit = match.groups()
        return int(num), unit
    return None, None


def _process_custom_unit(streamflow_data, custom_unit):
    """Process streamflow data with custom unit format like mm/3h."""
    custom_unit_pattern = re.compile(r"mm/(\d+)(h|d)")
    if custom_match := custom_unit_pattern.match(custom_unit):
        num, unit = custom_match.groups()
        if unit == "h":
            standard_unit = "mm/h"
            conversion_factor = int(num)
        elif unit == "d":
            standard_unit = "mm/d"
            conversion_factor = int(num)
        else:
            raise ValueError(f"Unsupported unit: {unit}")

        # Convert custom unit to standard unit
        if isinstance(streamflow_data, xr.Dataset):
            # For xarray, modify the data and attributes
            result = streamflow_data / conversion_factor
            result[list(result.keys())[0]].attrs["units"] = standard_unit
            return result
        else:
            # For numpy/pandas, just return the converted values
            return streamflow_data / conversion_factor, standard_unit
    else:
        # If it's not a custom unit format, return as is
        if isinstance(streamflow_data, xr.Dataset):
            result = streamflow_data.copy()
            result[list(result.keys())[0]].attrs["units"] = custom_unit
            return result
        else:
            return streamflow_data, custom_unit


def _get_unit_conversion_info(unit_str):
    """Get conversion information for a unit string.

    Returns:
        tuple: (standard_unit, conversion_factor) where conversion_factor
               is used to convert from standard unit to custom unit.
    """
    if not (match := re.match(r"mm/(\d+)(h|d)", unit_str)):
        # For standard units, no conversion needed
        return unit_str, 1
    num, unit = match.groups()
    if unit == "h":
        return "mm/h", int(num)
    elif unit == "d":
        return "mm/d", int(num)
    else:
        raise ValueError(f"Unsupported unit: {unit}")


def _get_actual_source_unit(streamflow_data, source_unit=None):
    """Determine the actual source unit from streamflow data.

    Parameters
    ----------
    streamflow_data : xarray.Dataset, pint.Quantity, numpy.ndarray,
                      pandas.DataFrame/Series
        The streamflow data to extract units from
    source_unit : str, optional
        Explicitly provided source unit that overrides data units

    Returns
    -------
    str or None
        The actual source unit string, or None if no unit information found
    """
    if source_unit is not None:
        return source_unit

    if isinstance(streamflow_data, xr.Dataset):
        streamflow_key = list(streamflow_data.keys())[0]
        # First check attrs for units
        if "units" in streamflow_data[streamflow_key].attrs:
            return streamflow_data[streamflow_key].attrs["units"]
        # Then check if it has pint units
        try:
            return str(streamflow_data[streamflow_key].pint.units)
        except (AttributeError, ValueError):
            return None
    elif isinstance(streamflow_data, pint.Quantity):
        return str(streamflow_data.units)
    else:
        # numpy array or pandas without units
        return None


def _normalize_unit(unit_str):
    """Normalize unit string for comparison (handle m3/s vs m^3/s and pint format)."""
    if not unit_str:
        return unit_str

    # Handle pint verbose format
    normalized = unit_str.replace("meter ** 3 / second", "m^3/s")
    normalized = normalized.replace("meter**3/second", "m^3/s")
    normalized = normalized.replace("cubic_meter / second", "m^3/s")
    normalized = normalized.replace("cubic_meter/second", "m^3/s")

    # Handle short format variations
    normalized = normalized.replace("m3/s", "m^3/s")
    normalized = normalized.replace("ft3/s", "ft^3/s")
    normalized = normalized.replace("ft**3/s", "ft^3/s")
    normalized = normalized.replace("cubic_foot / second", "ft^3/s")
    normalized = normalized.replace("cubic_foot/second", "ft^3/s")

    # Handle pint format for depth units
    normalized = normalized.replace("millimeter / day", "mm/d")
    normalized = normalized.replace("millimeter/day", "mm/d")
    normalized = normalized.replace("millimeter / hour", "mm/h")
    normalized = normalized.replace("millimeter/hour", "mm/h")

    return normalized


def _is_inverse_conversion(source_unit, target_unit):
    """Determine if this should be an inverse conversion based on units.

    Returns True if converting from depth units (mm/time) to volume units
    (m^3/s).
    Returns False if converting from volume units to depth units.
    """
    source_norm = _normalize_unit(source_unit) if source_unit else ""
    target_norm = _normalize_unit(target_unit)

    # Define unit patterns
    depth_pattern = re.compile(r"mm/(?:\d+)?[hd]?(?:ay|our)?$")
    volume_pattern = re.compile(r"(?:m\^?3|ft\^?3)/s$")

    source_is_depth = bool(depth_pattern.match(source_norm))
    source_is_volume = bool(volume_pattern.match(source_norm))
    target_is_depth = bool(depth_pattern.match(target_norm))
    target_is_volume = bool(volume_pattern.match(target_norm))

    if source_is_depth and target_is_volume:
        return True
    elif source_is_volume and target_is_depth:
        return False
    else:
        # If we can't determine from units, return None to indicate ambiguity
        return None


def _validate_inverse_consistency(source_unit, target_unit, inverse_param):
    """Validate that the inverse parameter is consistent with the units.

    Parameters
    ----------
    source_unit : str
        Source unit string
    target_unit : str
        Target unit string
    inverse_param : bool
        The inverse parameter provided by user

    Raises
    ------
    ValueError
        If inverse parameter is inconsistent with unit conversion direction
    """
    expected_inverse = _is_inverse_conversion(source_unit, target_unit)

    if expected_inverse is not None and expected_inverse != inverse_param:
        direction = "depth->volume" if expected_inverse else "volume->depth"
        raise ValueError(
            f"Inverse parameter ({inverse_param}) is inconsistent with unit "
            f"conversion direction. Converting from '{source_unit}' to "
            f"'{target_unit}' suggests {direction} conversion "
            f"(inverse={expected_inverse})."
        )


def streamflow_unit_conv(
    streamflow,
    area,
    target_unit="mm/d",
    inverse=False,
    source_unit=None,
    area_unit="km^2",
):
    """Convert the unit of streamflow data from m^3/s or ft^3/s to mm/xx(time) for a basin or inverse.

    Parameters
    ----------
    streamflow: xarray.Dataset, numpy.ndarray, pandas.DataFrame/Series, or pint.Quantity
        Streamflow data of each basin.
    area: xarray.Dataset, pint.Quantity, numpy.ndarray, pandas.DataFrame/Series
        Area of each basin. Can be with or without units.
    target_unit: str
        The unit to convert to.
    inverse: bool
        If True, convert the unit to m^3/s.
        If False, convert the unit to mm/day or mm/h.
    source_unit: str, optional
        The source unit of streamflow data. Use this when streamflow doesn't have
        unit information or when the unit is a custom format like 'mm/3h' that
        pint cannot recognize directly. If None, the function will try to get
        unit information from streamflow data attributes.
    area_unit: str, optional
        The unit of area data when area is provided without units (e.g., numpy array).
        Default is "km^2". Only used when area doesn't have unit information.

    Returns
    -------
    Converted data in the same type as the input streamflow.
    For numpy arrays, returns numpy array directly.
    """
    # Determine the actual source unit from data or parameter
    actual_source_unit = _get_actual_source_unit(streamflow, source_unit)

    # Normalize units for comparison
    source_normalized = _normalize_unit(actual_source_unit)
    target_normalized = _normalize_unit(target_unit)

    # Early return if source and target units are identical
    if source_normalized and source_normalized == target_normalized:
        return streamflow

    # Validate inverse parameter consistency with units
    if actual_source_unit:
        _validate_inverse_consistency(actual_source_unit, target_unit, inverse)

    # Get conversion information for target unit
    target_standard_unit, target_conversion_factor = _get_unit_conversion_info(
        target_unit
    )

    # Get conversion information for source unit if provided
    if source_unit:
        source_standard_unit, source_conversion_factor = _get_unit_conversion_info(
            source_unit
        )
    else:
        source_standard_unit, source_conversion_factor = None, 1

    # Regular expression to match units with numbers
    custom_unit_pattern = re.compile(r"mm/(\d+)(h|d)")

    # Function to handle the conversion for numpy and pandas
    def np_pd_conversion(streamflow, area, target_unit, inverse, conversion_factor):
        if not inverse:
            result = (streamflow / area).to(target_unit) * conversion_factor
        else:
            result = (streamflow * area).to(target_unit) / conversion_factor
        return result.magnitude

    # Handle xarray
    if isinstance(streamflow, xr.Dataset) and isinstance(area, xr.Dataset):
        # Check for units in attrs first, then try pint-xarray units
        streamflow_key = list(streamflow.keys())[0]
        streamflow_units = streamflow[streamflow_key].attrs.get("units", None)

        # Check if streamflow has pint units
        has_pint_units = False
        try:
            # Check if data already has pint units
            streamflow[streamflow_key].pint.units
            has_pint_units = True
        except (AttributeError, ValueError):
            has_pint_units = False

        # Handle source_unit parameter
        if source_unit is not None:
            # Process custom units with source conversion factor
            if isinstance(streamflow, xr.Dataset):
                # For xarray, convert custom unit to standard unit
                streamflow_processed = streamflow / source_conversion_factor
                key = list(streamflow_processed.keys())[0]
                streamflow_processed[key].attrs["units"] = source_standard_unit
            else:
                streamflow_processed = streamflow
        elif streamflow_units is None and not has_pint_units:
            raise ValueError(
                "streamflow has no unit information. "
                "Please provide source_unit parameter."
            )
        else:
            streamflow_processed = streamflow

        if not inverse:
            if not (
                custom_unit_pattern.match(target_unit)
                or re.match(r"mm/(?!\d)", target_unit)
            ):
                raise ValueError(
                    "target_unit should be a valid unit like 'mm/d', 'mm/day', 'mm/h', 'mm/hour', 'mm/3h', 'mm/5d'"
                )

            try:
                q = streamflow_processed.pint.quantify()
            except Exception as e:
                # If pint-xarray is not available, provide a helpful error message
                if "no attribute 'pint'" in str(e):
                    raise ValueError(
                        "pint-xarray extension is not available. "
                        "Please install pint-xarray or provide explicit source_unit parameter."
                    )
                elif source_unit is None:
                    raise ValueError(
                        "Failed to quantify streamflow units. "
                        f"Please provide source_unit parameter. Error: {e}"
                    )
                else:
                    raise ValueError(
                        f"Failed to quantify streamflow with source_unit "
                        f"'{source_unit}'. Error: {e}"
                    )

            a = area.pint.quantify()
            r = q[list(q.keys())[0]] / a[list(a.keys())[0]]
            # result = r.pint.to(target_unit).to_dataset(name=list(q.keys())[0])
            result = (
                r.pint.to(target_standard_unit) * target_conversion_factor
            ).to_dataset(name=list(q.keys())[0])
            # Manually set the unit attribute to the custom unit
            result_ = result.pint.dequantify()
            result_[list(result_.keys())[0]].attrs["units"] = target_unit
            return result_
        else:
            # For inverse conversion
            if target_unit not in ["m^3/s", "m3/s"]:
                raise ValueError("target_unit should be 'm^3/s'")

            # Handle source_unit for inverse conversion
            if source_unit is not None:
                # Process custom units with source conversion factor
                streamflow_processed = streamflow / source_conversion_factor
                key = list(streamflow_processed.keys())[0]
                streamflow_processed[key].attrs["units"] = source_standard_unit
            else:
                streamflow_units = streamflow[list(streamflow.keys())[0]].attrs.get(
                    "units", None
                )

                # Check if streamflow has pint units for inverse conversion
                has_pint_units_inverse = False
                try:
                    streamflow[list(streamflow.keys())[0]].pint.units
                    has_pint_units_inverse = True
                except (AttributeError, ValueError):
                    has_pint_units_inverse = False

                if streamflow_units:
                    if custom_match := custom_unit_pattern.match(streamflow_units):
                        num, unit = custom_match.groups()
                        if unit == "h":
                            temp_standard_unit = "mm/h"
                            temp_conversion_factor = int(num)
                        elif unit == "d":
                            temp_standard_unit = "mm/d"
                            temp_conversion_factor = int(num)
                        # Convert custom unit to standard unit
                        r_ = streamflow / temp_conversion_factor
                        r_[list(r_.keys())[0]].attrs["units"] = temp_standard_unit
                        streamflow_processed = r_
                    else:
                        streamflow_processed = streamflow
                elif has_pint_units_inverse:
                    # Data has pint units, use as is
                    streamflow_processed = streamflow
                else:
                    raise ValueError(
                        "streamflow has no unit information. "
                        "Please provide source_unit parameter."
                    )

            try:
                r = streamflow_processed.pint.quantify()
            except Exception as e:
                raise ValueError(
                    f"Failed to quantify streamflow units for inverse conversion. "
                    f"Error: {e}"
                )

            a = area.pint.quantify()
            q = r[list(r.keys())[0]] * a[list(a.keys())[0]]
            result = q.pint.to(target_unit).to_dataset(name=list(r.keys())[0])
            # dequantify to get normal xr_dataset
            return result.pint.dequantify()

    # Handle numpy and pandas
    elif isinstance(streamflow, pint.Quantity) and isinstance(area, pint.Quantity):
        if type(streamflow.magnitude) not in [np.ndarray, pd.DataFrame, pd.Series]:
            raise TypeError(
                "Input streamflow must be xarray.Dataset, or pint.Quantity "
                "wrapping numpy.ndarray, or pandas.DataFrame/Series"
            )
        if type(area.magnitude) != type(streamflow.magnitude):
            raise TypeError("streamflow and area must be the same type")
        return np_pd_conversion(
            streamflow, area, target_standard_unit, inverse, target_conversion_factor
        )

    # Handle numpy and pandas without units (requires source_unit)
    elif isinstance(streamflow, (np.ndarray, pd.DataFrame, pd.Series)):
        if source_unit is None:
            raise ValueError(
                "streamflow data has no unit information. "
                "Please provide source_unit parameter."
            )

        # Process custom unit if needed
        streamflow_processed = streamflow / source_conversion_factor
        processed_unit = source_standard_unit

        try:
            # Create pint quantity for streamflow
            streamflow_qty = streamflow_processed * ureg(processed_unit)
        except Exception as e:
            raise ValueError(
                f"Failed to create quantity with unit '{processed_unit}'. Error: {e}"
            )

        # Handle area with or without units
        if isinstance(area, pint.Quantity):
            area_qty = area
        elif isinstance(area, (np.ndarray, pd.DataFrame, pd.Series)):
            # Area has no units, use area_unit parameter
            try:
                area_qty = area * ureg(area_unit)
            except Exception as e:
                raise ValueError(
                    f"Failed to create quantity with unit '{area_unit}'. Error: {e}"
                )
        else:
            raise TypeError(
                "area must be pint.Quantity, numpy.ndarray, or pandas.DataFrame/Series"
            )

        result = np_pd_conversion(
            streamflow_qty,
            area_qty,
            target_standard_unit,
            inverse,
            target_conversion_factor,
        )

        # For numpy/pandas input, return numpy array directly (not pint.Quantity)
        return result

    else:
        raise TypeError(
            "Input streamflow must be xarray.Dataset, pint.Quantity wrapping "
            "numpy.ndarray/pandas.DataFrame/Series, or numpy.ndarray/pandas."
            "DataFrame/Series with source_unit parameter"
        )


def detect_time_interval(
    time_series: Union[pd.DatetimeIndex, list, np.ndarray],
) -> str:
    """
    Automatically detect the time interval of a time series.

    Parameters
    ----------
    time_series : pd.DatetimeIndex, list, or np.ndarray
        Time series data with datetime information

    Returns
    -------
    str
        Detected time interval in format suitable for unit conversion
        (e.g., "1h", "3h", "1d")

    Examples
    --------
    >>> import pandas as pd
    >>> time_index = pd.date_range("2024-01-01", periods=8, freq="3h")
    >>> detect_time_interval(time_index)
    '3h'
    """
    if isinstance(time_series, list):
        time_series = pd.to_datetime(time_series)
    elif isinstance(time_series, np.ndarray):
        time_series = pd.to_datetime(time_series)
    elif not isinstance(time_series, pd.DatetimeIndex):
        time_series = pd.DatetimeIndex(time_series)

    if len(time_series) < 2:
        raise ValueError(
            "Time series must have at least 2 time points to detect interval"
        )

    # Calculate time differences
    time_diffs = time_series[1:] - time_series[:-1]

    # Get the most common time difference (mode)
    most_common_diff = time_diffs.value_counts().index[0]

    # Convert to hours and days
    total_seconds = most_common_diff.total_seconds()
    hours = total_seconds / 3600
    days = total_seconds / (3600 * 24)

    # Determine the appropriate unit
    if hours == int(hours) and hours < 24:
        return f"{int(hours)}h"
    elif days == int(days):
        return f"{int(days)}d"
    else:
        # For non-integer hours, round to nearest hour
        rounded_hours = round(hours)
        return f"{rounded_hours}h"


def get_time_interval_info(time_interval: str) -> Tuple[int, str]:
    """
    Parse time interval string to extract number and unit.

    Parameters
    ----------
    time_interval : str
        Time interval string (e.g., "1h", "3h", "1d")

    Returns
    -------
    tuple
        (number, unit) where number is int and unit is str

    Examples
    --------
    >>> get_time_interval_info("3h")
    (3, 'h')
    >>> get_time_interval_info("1d")
    (1, 'd')
    """
    match = re.match(r"^(\d+)([hd])$", time_interval)
    if not match:
        raise ValueError(f"Invalid time interval format: {time_interval}")

    number, unit = match.groups()
    return int(number), unit


def validate_unit_compatibility(source_unit: str, target_unit: str) -> bool:
    """
    Check if two units are compatible for conversion.

    Parameters
    ----------
    source_unit : str
        Source unit string
    target_unit : str
        Target unit string

    Returns
    -------
    bool
        True if units are compatible for conversion

    Examples
    --------
    >>> validate_unit_compatibility("mm/3h", "m^3/s")
    True
    >>> validate_unit_compatibility("mm/h", "celsius")
    False
    """
    # Define unit categories
    depth_units = re.compile(r"mm/\d*[hd]")
    volume_units = re.compile(r"m\^?3/s")

    source_is_depth = bool(depth_units.match(source_unit))
    source_is_volume = bool(volume_units.match(source_unit))
    target_is_depth = bool(depth_units.match(target_unit))
    target_is_volume = bool(volume_units.match(target_unit))

    # Compatible if both are depth units, both are volume units, or one of each
    return (source_is_depth or source_is_volume) and (
        target_is_depth or target_is_volume
    )
