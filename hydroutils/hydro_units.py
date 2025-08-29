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
    normalized = normalized.replace("foot^3/s", "ft^3/s")
    normalized = normalized.replace("foot**3/s", "ft^3/s")
    normalized = normalized.replace("foot3/s", "ft^3/s")
    normalized = normalized.replace("cubic_foot / second", "ft^3/s")
    normalized = normalized.replace("cubic_foot/second", "ft^3/s")

    # Handle pint format for depth units
    normalized = normalized.replace("millimeter / day", "mm/d")
    normalized = normalized.replace("millimeter/day", "mm/d")
    normalized = normalized.replace("mm/day", "mm/d")
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


def _detect_data_unit(data, source_unit=None):
    """Detect and validate the unit of streamflow data.

    Parameters
    ----------
    data : numpy.ndarray, pandas.Series, pandas.DataFrame, or xarray.Dataset
        Input data to detect units from
    source_unit : str, optional
        Explicitly provided source unit

    Returns
    -------
    str
        The detected unit string

    Raises
    ------
    ValueError
        If no unit can be detected and source_unit is not provided
        If source_unit conflicts with detected data units
    """
    detected_unit = None

    # Try to detect unit from data
    if isinstance(data, xr.Dataset):
        # Get first data variable key
        data_key = list(data.keys())[0]

        # Check attrs for units first
        if "units" in data[data_key].attrs:
            detected_unit = data[data_key].attrs["units"]
        else:
            # Check if the data itself is a pint.Quantity
            if isinstance(data[data_key].data, pint.Quantity):
                detected_unit = str(data[data_key].data.units)
                detected_unit = _normalize_unit(detected_unit)
            else:
                # Try pint-xarray units
                try:
                    detected_unit = str(data[data_key].pint.units)
                except (AttributeError, ValueError):
                    detected_unit = None

    elif isinstance(data, pint.Quantity):
        detected_unit = str(data.units)
        # Normalize pint unit strings for consistency
        detected_unit = _normalize_unit(detected_unit)
    elif hasattr(data, "attrs") and "units" in data.attrs:
        # For pandas with attrs
        detected_unit = data.attrs["units"]

    # Validate consistency if both detected and provided
    if detected_unit and source_unit:
        if _normalize_unit(detected_unit) != _normalize_unit(source_unit):
            raise ValueError(
                f"Provided source_unit '{source_unit}' conflicts with detected "
                f"data unit '{detected_unit}'"
            )

    # Determine final unit
    final_unit = source_unit or detected_unit

    if not final_unit:
        raise ValueError(
            "No unit information found in data. Please provide source_unit parameter."
        )

    return final_unit


def _detect_area_unit(area, area_unit="km^2"):
    """Detect and validate the unit of area data.

    Parameters
    ----------
    area : numpy.ndarray, pandas.Series, pandas.DataFrame, xarray.Dataset, or pint.Quantity
        Input area data
    area_unit : str, optional
        Default area unit when no units are detected. Default is "km^2".

    Returns
    -------
    str
        The detected or default area unit
    """
    detected_unit = None

    # Try to detect unit from area data
    if isinstance(area, xr.Dataset):
        area_key = list(area.keys())[0]
        if "units" in area[area_key].attrs:
            detected_unit = area[area_key].attrs["units"]
        else:
            try:
                detected_unit = str(area[area_key].pint.units)
            except (AttributeError, ValueError):
                detected_unit = None
    elif isinstance(area, pint.Quantity):
        detected_unit = str(area.units)
    elif hasattr(area, "attrs") and "units" in area.attrs:
        detected_unit = area.attrs["units"]

    # Use detected unit or fallback to provided area_unit
    return detected_unit or area_unit


def _determine_conversion_direction(source_unit, target_unit):
    """Determine if conversion is from depth to volume units or vice versa.

    Parameters
    ----------
    source_unit : str
        Source unit string
    target_unit : str
        Target unit string

    Returns
    -------
    bool
        True if converting from depth units to volume units
        False if converting from volume units to depth units

    Raises
    ------
    ValueError
        If units are incompatible for conversion
    """
    source_norm = _normalize_unit(source_unit)
    target_norm = _normalize_unit(target_unit)

    depth_patterns = [
        r"mm/\d*[hH](?:our|r)?s?$",  # mm/h, mm/3h, mm/24h, mm/hour, mm/Hr
        r"mm/\d*[dD](?:ay)?s?$",  # mm/d, mm/1d, mm/day, mm/Day
        r"mm/\d*[mM](?:in)?s?$",  # mm/m, mm/30m, mm/min
        r"mm/\d*[sS](?:ec)?s?$",  # mm/s, mm/sec
        r"in/\d*[hH](?:our|r)?s?$",  # in/h, in/hour
        r"in/\d*[dD](?:ay)?s?$",  # in/d, in/day
    ]

    volume_patterns = [
        r"m(?:\*\*3|\^3|3)/s$",  # m**3/s, m^3/s, m3/s
        r"ft(?:\*\*3|\^3|3)/s$",  # ft**3/s, ft^3/s, ft3/s
        r"l/s$",  # l/s
        r"gal/s$",  # gal/s
    ]

    source_is_depth = any(re.match(pattern, source_norm) for pattern in depth_patterns)
    source_is_volume = any(
        re.match(pattern, source_norm) for pattern in volume_patterns
    )
    target_is_depth = any(re.match(pattern, target_norm) for pattern in depth_patterns)
    target_is_volume = any(
        re.match(pattern, target_norm) for pattern in volume_patterns
    )

    # Validate compatibility
    if not (
        (source_is_depth or source_is_volume) and (target_is_depth or target_is_volume)
    ):
        raise ValueError(
            f"Incompatible units for conversion: '{source_unit}' to '{target_unit}'"
        )

    if source_is_depth and target_is_volume:
        return True  # depth to volume
    elif source_is_volume and target_is_depth:
        return False  # volume to depth
    else:
        # Same type conversion (depth to depth or volume to volume)
        return None


def _perform_conversion(
    data, area, source_unit, area_unit, target_unit, is_depth_to_volume
):
    """Perform the actual unit conversion based on data type.

    Parameters
    ----------
    data : numpy.ndarray, pandas.Series, pandas.DataFrame, or xarray.Dataset
        Input streamflow data
    area : numpy.ndarray, pandas.Series, pandas.DataFrame, xarray.Dataset, or pint.Quantity
        Area data
    source_unit : str
        Source unit of data
    area_unit : str
        Unit of area data
    target_unit : str
        Target unit for conversion
    is_depth_to_volume : bool or None
        Conversion direction. None for same-type conversions.

    Returns
    -------
    Converted data in same format as input
    """
    # Handle custom units (mm/3h, mm/5d, etc.)
    source_standard_unit, source_factor = _get_unit_conversion_info(source_unit)
    target_standard_unit, target_factor = _get_unit_conversion_info(target_unit)

    # Dispatch to appropriate conversion method based on data type
    if isinstance(data, xr.Dataset):
        return _convert_xarray(
            data,
            area,
            source_standard_unit,
            source_factor,
            area_unit,
            target_standard_unit,
            target_factor,
            target_unit,
            is_depth_to_volume,
        )
    elif isinstance(data, pint.Quantity):
        return _convert_pint_quantity(
            data,
            area,
            source_standard_unit,
            source_factor,
            area_unit,
            target_standard_unit,
            target_factor,
            is_depth_to_volume,
        )
    elif isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
        return _convert_numpy_pandas(
            data,
            area,
            source_standard_unit,
            source_factor,
            area_unit,
            target_standard_unit,
            target_factor,
            is_depth_to_volume,
        )
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _convert_xarray(
    data,
    area,
    source_unit,
    source_factor,
    area_unit,
    target_unit,
    target_factor,
    target_unit_str,
    is_depth_to_volume,
):
    """Convert units of an xarray Dataset with support for pint-xarray.

    This function handles unit conversion for xarray Datasets, attempting to use
    pint-xarray for unit-aware calculations when available, with a fallback to
    manual conversion using pint directly.

    Args:
        data (xr.Dataset): Input dataset containing streamflow data.
        area (Union[xr.Dataset, np.ndarray]): Basin area data.
        source_unit (str): Source unit (standard form, e.g., 'mm/h').
        source_factor (int): Factor to convert from custom to standard unit.
        area_unit (str): Unit of the area data.
        target_unit (str): Target unit (standard form).
        target_factor (int): Factor to convert from standard to custom unit.
        target_unit_str (str): Full target unit string for metadata.
        is_depth_to_volume (bool): True for depth->volume, False for volume->depth.

    Returns:
        xr.Dataset: Converted dataset with updated units in metadata.

    Note:
        - Tries pint-xarray first for unit-aware calculations
        - Falls back to manual conversion with pint if pint-xarray unavailable
        - Preserves dataset structure and coordinates
        - Updates unit metadata in dataset attributes
        - Handles both depth->volume and volume->depth conversions

    Example:
        >>> import xarray as xr
        >>> data = xr.Dataset({'flow': ('time', [1, 2, 3])})
        >>> data.flow.attrs['units'] = 'mm/h'
        >>> area = xr.Dataset({'area': 1000})  # km²
        >>> result = _convert_xarray(data, area, 'mm/h', 1,
        ...                         'km^2', 'm^3/s', 1, 'm^3/s', True)
        >>> print(result.flow.attrs['units'])
        'm^3/s'
    """
    data_key = list(data.keys())[0]

    # Try pint-xarray first, fallback to manual conversion if not available
    try:
        # Check if pint-xarray is available
        if hasattr(data[data_key], "pint"):
            data_qty = data.pint.quantify()

            if isinstance(area, xr.Dataset):
                area_qty = area.pint.quantify()
            else:
                # Convert area to xr.Dataset with units
                area_qty = xr.Dataset(
                    {
                        data_key: (
                            ["time"] if "time" in data.dims else list(data.dims)[:1],
                            area,
                        )
                    }
                )
                area_qty[data_key].attrs["units"] = area_unit
                area_qty = area_qty.pint.quantify()

            # Perform conversion using pint-xarray
            data_var = data_qty[data_key] / source_factor
            area_var = area_qty[list(area_qty.keys())[0]]

            if is_depth_to_volume is True:
                result_qty = data_var * area_var
            elif is_depth_to_volume is False:
                result_qty = data_var / area_var
            else:
                result_qty = data_var

            # Convert to target unit
            result_qty = result_qty.pint.to(target_unit) * target_factor
            result_dataset = result_qty.to_dataset(name=data_key)

            # Dequantify and set final unit
            result_final = result_dataset.pint.dequantify()
            result_final[data_key].attrs["units"] = target_unit_str

            return result_final

    except (AttributeError, ImportError):
        # Fallback to manual conversion without pint-xarray
        pass

    # Manual conversion without pint-xarray
    data_values = data[data_key].values / source_factor

    if isinstance(area, xr.Dataset):
        area_key = list(area.keys())[0]
        area_values = area[area_key].values
    else:
        area_values = area

    # Create pint quantities for conversion
    try:
        data_qty = data_values * ureg(source_unit)
        area_qty = area_values * ureg(area_unit)

        # Perform conversion
        if is_depth_to_volume is True:
            result_qty = data_qty * area_qty
        elif is_depth_to_volume is False:
            result_qty = data_qty / area_qty
        else:
            result_qty = data_qty

        # Convert to target unit
        converted_qty = result_qty.to(ureg(target_unit))
        result_values = converted_qty.magnitude * target_factor

        # Create result dataset
        result = data.copy()
        result[data_key] = result[data_key].copy()
        result[data_key].values = result_values
        result[data_key].attrs["units"] = target_unit_str

        return result

    except Exception as e:
        raise ValueError(f"Failed to convert xarray data: {e}")


def _convert_pint_quantity(
    data,
    area,
    source_unit,
    source_factor,
    area_unit,
    target_unit,
    target_factor,
    is_depth_to_volume,
):
    """Convert units of a pint.Quantity object.

    This function handles unit conversion for pint.Quantity objects, which
    already have unit information attached. It supports both depth->volume
    and volume->depth conversions.

    Args:
        data (pint.Quantity): Input streamflow data with units.
        area (Union[pint.Quantity, np.ndarray]): Basin area data.
        source_unit (str): Source unit (standard form, e.g., 'mm/h').
        source_factor (int): Factor to convert from custom to standard unit.
        area_unit (str): Unit of the area data if not a pint.Quantity.
        target_unit (str): Target unit (standard form).
        target_factor (int): Factor to convert from standard to custom unit.
        is_depth_to_volume (bool): True for depth->volume, False for volume->depth.

    Returns:
        np.ndarray: Converted data values (not a pint.Quantity).

    Note:
        - Extracts magnitude and units from pint.Quantity objects
        - Handles area data as either pint.Quantity or raw values
        - Performs unit-aware calculations using pint
        - Returns raw numpy array to match interface
        - Applies custom unit factors for non-standard units

    Example:
        >>> import pint
        >>> ureg = pint.UnitRegistry()
        >>> data = [1, 2, 3] * ureg('mm/h')
        >>> area = 1000 * ureg('km^2')
        >>> result = _convert_pint_quantity(data, area, 'mm/h', 1,
        ...                                'km^2', 'm^3/s', 1, True)
        >>> print(result)  # Values in m^3/s
        array([277.77..., 555.55..., 833.33...])
    """
    # Extract values and handle source factor
    data_values = data.magnitude / source_factor
    data_units = str(data.units)

    # Handle area - could be pint.Quantity or numpy/pandas
    if isinstance(area, pint.Quantity):
        area_values = area.magnitude
        area_unit_for_calc = str(area.units)
    else:
        area_values = area
        area_unit_for_calc = area_unit

    # Create pint quantities for conversion
    try:
        data_qty = data_values * ureg(source_unit)
        area_qty = area_values * ureg(area_unit_for_calc)

        # Perform conversion based on direction
        if is_depth_to_volume is True:
            result_qty = data_qty * area_qty
        elif is_depth_to_volume is False:
            result_qty = data_qty / area_qty
        else:
            # Same type conversion
            result_qty = data_qty

        # Convert to target unit
        converted_qty = result_qty.to(ureg(target_unit))
        final_result = converted_qty.magnitude * target_factor

        # Return as numpy array (not pint.Quantity) to match interface
        return final_result

    except Exception as e:
        raise ValueError(f"Failed to convert pint.Quantity data: {e}")


def _convert_numpy_pandas(
    data,
    area,
    source_unit,
    source_factor,
    area_unit,
    target_unit,
    target_factor,
    is_depth_to_volume,
):
    """Convert units for numpy arrays and pandas objects.

    This function handles unit conversion for numpy arrays, pandas Series, and
    pandas DataFrames. It preserves the input data structure while performing
    unit conversions using pint.

    Args:
        data (Union[np.ndarray, pd.Series, pd.DataFrame]): Input data.
        area (Union[np.ndarray, pd.Series, pd.DataFrame, pint.Quantity]): Basin area.
        source_unit (str): Source unit (standard form, e.g., 'mm/h').
        source_factor (int): Factor to convert from custom to standard unit.
        area_unit (str): Unit of the area data if not a pint.Quantity.
        target_unit (str): Target unit (standard form).
        target_factor (int): Factor to convert from standard to custom unit.
        is_depth_to_volume (bool): True for depth->volume, False for volume->depth.

    Returns:
        Union[np.ndarray, pd.Series, pd.DataFrame]: Converted data in same format
            as input.

    Note:
        - Preserves pandas index and column structure
        - Handles area data in various formats
        - Uses pint for unit-aware calculations
        - Applies custom unit factors for non-standard units
        - Returns same type as input data

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> # NumPy array example
        >>> data = np.array([1, 2, 3])  # mm/h
        >>> area = np.array([1000])  # km²
        >>> result = _convert_numpy_pandas(data, area, 'mm/h', 1,
        ...                               'km^2', 'm^3/s', 1, True)
        >>> print(result)  # Values in m^3/s
        array([277.77..., 555.55..., 833.33...])
        
        >>> # Pandas Series example
        >>> data = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        >>> result = _convert_numpy_pandas(data, area, 'mm/h', 1,
        ...                               'km^2', 'm^3/s', 1, True)
        >>> print(result.index)
        Index(['a', 'b', 'c'], dtype='object')
    """
    # Extract values for computation, preserve data structure
    is_pandas = isinstance(data, (pd.Series, pd.DataFrame))
    data_index = None
    data_columns = None

    if isinstance(data, pd.Series):
        data_values = data.values / source_factor
        data_index = data.index
    elif isinstance(data, pd.DataFrame):
        data_values = data.values / source_factor
        data_index = data.index
        data_columns = data.columns
    else:
        data_values = data / source_factor

    # Handle area values
    if isinstance(area, pint.Quantity):
        area_values = area.magnitude
        area_unit_for_calc = str(area.units)
    elif isinstance(area, (pd.Series, pd.DataFrame)):
        area_values = area.values
        area_unit_for_calc = area_unit
    else:
        area_values = area
        area_unit_for_calc = area_unit

    # Create pint quantities for conversion
    try:
        data_qty = data_values * ureg(source_unit)
        area_qty = area_values * ureg(area_unit_for_calc)

        # Perform conversion
        if is_depth_to_volume is True:
            result_qty = data_qty * area_qty
        elif is_depth_to_volume is False:
            result_qty = data_qty / area_qty
        else:
            result_qty = data_qty

        # Convert to target unit
        converted_qty = result_qty.to(ureg(target_unit))
        result_values = converted_qty.magnitude * target_factor

        # Reconstruct pandas structure if needed
        if isinstance(data, pd.Series):
            return pd.Series(result_values, index=data_index)
        elif isinstance(data, pd.DataFrame):
            return pd.DataFrame(result_values, index=data_index, columns=data_columns)
        else:
            return result_values

    except Exception as e:
        raise ValueError(f"Failed to convert numpy/pandas data: {e}")


def streamflow_unit_conv(
    data,
    area,
    target_unit,
    source_unit=None,
    area_unit="km^2",
):
    """Convert streamflow data units between depth units (mm/time) and volume units (m³/s).

    This function automatically detects conversion direction based on source and target units,
    removing the need for an explicit inverse parameter.

    Parameters
    ----------
    data : numpy.ndarray, pandas.Series, pandas.DataFrame, or xarray.Dataset
        Streamflow data. Can include unit information in attributes (xarray) or
        requires source_unit parameter for numpy/pandas data.
    area : numpy.ndarray, pandas.Series, pandas.DataFrame, xarray.Dataset, or pint.Quantity
        Basin area data. Units will be detected from data attributes or pint units.
        If no units detected, area_unit parameter will be used.
    target_unit : str
        Target unit for conversion. Examples: "mm/d", "mm/h", "mm/3h", "m^3/s".
    source_unit : str, optional
        Source unit of streamflow data. Required if data has no unit information.
        If provided and data has units, they must match or ValueError is raised.
    area_unit : str, optional
        Unit for area when area data has no unit information. Default is "km^2".

    Returns
    -------
    Converted data in the same type as input data.
    Unit information is preserved in xarray attributes when applicable.

    Raises
    ------
    ValueError
        If no unit information can be determined for data or area.
        If source_unit conflicts with detected data units.
        If units are incompatible for conversion.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Convert m³/s to mm/day
    >>> flow = np.array([10.5, 15.2, 8.1])
    >>> basin_area = np.array([1000])  # km²
    >>> result = streamflow_unit_conv(flow, basin_area, "mm/d", source_unit="m^3/s")

    >>> # Convert mm/h to m³/s
    >>> flow_mm = np.array([2.1, 3.5, 1.8])
    >>> result = streamflow_unit_conv(flow_mm, basin_area, "m^3/s", source_unit="mm/h")
    """
    # Step 1: Detect and validate source unit
    detected_source_unit = _detect_data_unit(data, source_unit)

    # Step 2: Detect and validate area unit
    detected_area_unit = _detect_area_unit(area, area_unit)

    # Step 3: Determine conversion direction and validate compatibility
    is_depth_to_volume = _determine_conversion_direction(
        detected_source_unit, target_unit
    )

    # Step 4: Early return if no conversion needed
    if _normalize_unit(detected_source_unit) == _normalize_unit(target_unit):
        return data

    # Step 5: Perform the actual conversion based on data type
    return _perform_conversion(
        data,
        area,
        detected_source_unit,
        detected_area_unit,
        target_unit,
        is_depth_to_volume,
    )


def detect_time_interval(
    time_series: Union[pd.DatetimeIndex, list, np.ndarray],
) -> str:
    """Detect the time interval between points in a time series.

    This function analyzes a time series to determine the most common time
    interval between consecutive points. It handles various input formats and
    converts the interval to a standardized string format.

    Args:
        time_series (Union[pd.DatetimeIndex, list, np.ndarray]): Time series
            data containing datetime information. Can be:
            - pandas DatetimeIndex
            - List of datetime-like objects
            - NumPy array of datetime-like objects

    Returns:
        str: Detected time interval in format suitable for unit conversion:
            - For hourly data: "Nh" where N is number of hours (e.g., "3h")
            - For daily data: "Nd" where N is number of days (e.g., "1d")

    Raises:
        ValueError: If time series has fewer than 2 points.

    Note:
        - Uses most common time difference (mode) for irregular intervals
        - Rounds non-integer hours to nearest hour
        - Prefers hours for intervals < 24h, days for intervals ≥ 24h
        - Automatically converts various datetime formats to pandas DatetimeIndex

    Example:
        >>> import pandas as pd
        >>> # Regular 3-hourly data
        >>> time_index = pd.date_range("2024-01-01", periods=8, freq="3h")
        >>> detect_time_interval(time_index)
        '3h'
        
        >>> # Daily data
        >>> dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        >>> detect_time_interval(dates)
        '1d'
        
        >>> # Mixed intervals (most common is 6h)
        >>> times = pd.to_datetime([
        ...     "2024-01-01 00:00",
        ...     "2024-01-01 06:00",
        ...     "2024-01-01 12:00",
        ...     "2024-01-01 18:00"
        ... ])
        >>> detect_time_interval(times)
        '6h'
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
    """Parse a time interval string into its numeric value and unit.

    This function extracts the numeric value and unit from a time interval
    string using regular expressions. It supports hourly and daily intervals
    in a standardized format.

    Args:
        time_interval (str): Time interval string in format "Nh" or "Nd" where
            N is a positive integer. Examples: "1h", "3h", "1d", "5d".

    Returns:
        Tuple[int, str]: Two-element tuple containing:
            - number (int): The numeric value from the interval
            - unit (str): The unit, either 'h' for hours or 'd' for days

    Raises:
        ValueError: If time_interval doesn't match expected format.

    Note:
        - Only supports hours ('h') and days ('d') units
        - Number must be a positive integer
        - Format is case-sensitive ('h' and 'd' must be lowercase)
        - No spaces allowed in the interval string

    Example:
        >>> # Hourly intervals
        >>> get_time_interval_info("3h")
        (3, 'h')
        >>> get_time_interval_info("24h")
        (24, 'h')
        
        >>> # Daily intervals
        >>> get_time_interval_info("1d")
        (1, 'd')
        >>> get_time_interval_info("7d")
        (7, 'd')
        
        >>> # Invalid format raises error
        >>> get_time_interval_info("3hours")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: Invalid time interval format: 3hours
    """
    match = re.match(r"^(\d+)([hd])$", time_interval)
    if not match:
        raise ValueError(f"Invalid time interval format: {time_interval}")

    number, unit = match.groups()
    return int(number), unit


def validate_unit_compatibility(source_unit: str, target_unit: str) -> bool:
    """Check if two hydrological units can be converted between each other.

    This function determines whether two units are compatible for hydrological
    unit conversion. It supports depth units (mm/time) and volume units (m³/s),
    and checks if the conversion between them is possible.

    Args:
        source_unit (str): Source unit string. Examples:
            - Depth units: "mm/h", "mm/3h", "mm/d", "in/d"
            - Volume units: "m^3/s", "ft^3/s", "l/s"
        target_unit (str): Target unit string (same format as source_unit).

    Returns:
        bool: True if units are compatible for conversion, False otherwise.

    Note:
        - Supports various time intervals for depth units
        - Recognizes multiple formats for volume units
        - Case-sensitive unit matching
        - Compatible conversions:
            - depth -> volume (e.g., mm/h -> m³/s)
            - volume -> depth (e.g., m³/s -> mm/d)
            - depth -> depth (e.g., mm/h -> mm/d)
            - volume -> volume (e.g., m³/s -> ft³/s)

    Example:
        >>> # Compatible conversions
        >>> validate_unit_compatibility("mm/3h", "m^3/s")
        True
        >>> validate_unit_compatibility("m^3/s", "mm/d")
        True
        >>> validate_unit_compatibility("mm/h", "mm/d")
        True
        
        >>> # Incompatible conversions
        >>> validate_unit_compatibility("mm/h", "celsius")
        False
        >>> validate_unit_compatibility("m^3/s", "kg/m^3")
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
