# hydro_units

The `hydro_units` module provides comprehensive unit conversion functionality for hydrological data.

## Core Functions

### streamflow_unit_conv

```python
def streamflow_unit_conv(
    streamflow_data: Union[xr.Dataset, pint.Quantity, np.ndarray, pd.DataFrame, pd.Series],
    source_unit: str = None,
    target_unit: str = "m³/s",
    basin_area: float = None,
    time_interval: str = None
) -> Union[xr.Dataset, pint.Quantity, np.ndarray, pd.DataFrame, pd.Series]
```

Converts streamflow data between different units (depth-based to volume-based and vice versa).

**Example:**
```python
import hydroutils as hu
import numpy as np

# Convert from mm/h to m³/s
flow_mmh = np.array([10, 20, 15])  # mm/h
flow_m3s = hu.streamflow_unit_conv(
    flow_mmh, 
    source_unit="mm/h", 
    target_unit="m³/s",
    basin_area=1000  # km²
)
```

### detect_time_interval

```python
def detect_time_interval(time_series: Union[pd.Series, np.ndarray, list]) -> str
```

Automatically detects the time interval from a time series.

### get_time_interval_info

```python
def get_time_interval_info(time_interval: str) -> dict
```

Returns detailed information about a time interval.

### validate_unit_compatibility

```python
def validate_unit_compatibility(unit1: str, unit2: str) -> bool
```

Validates if two units are compatible for conversion.

## API Reference

::: hydroutils.hydro_units
