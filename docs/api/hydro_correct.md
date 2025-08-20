# hydro_correct

The `hydro_correct` module provides interactive correction algorithms for hydrological forecasting, implementing five-point quadratic smoothing and cubic spline interpolation based on Zhang Silong's 2006 paper.

## Classes

### HydrographCorrector

```python
class HydrographCorrector:
    def __init__(self, time_points: np.ndarray, discharge_values: np.ndarray):
        """Initialize the corrector.

        Args:
            time_points (np.ndarray): Array of time points.
            discharge_values (np.ndarray): Array of discharge values.
        """
```

A class for interactive correction of flood hydrographs using five-point quadratic smoothing and cubic spline interpolation.

**Methods:**

#### five_point_smooth

```python
def five_point_smooth(self, discharge_values: np.ndarray) -> np.ndarray:
    """Apply five-point quadratic smoothing to discharge data.

    Args:
        discharge_values (np.ndarray): Array of discharge values.

    Returns:
        np.ndarray: Smoothed discharge values.

    Raises:
        ValueError: If input data length doesn't match initialization length.
    """
```

#### cubic_spline_interpolation

```python
def cubic_spline_interpolation(
    self, 
    x: np.ndarray, 
    y: np.ndarray, 
    x_new: Optional[np.ndarray] = None
) -> np.ndarray:
    """Perform cubic spline interpolation.

    Args:
        x (np.ndarray): Original time points.
        y (np.ndarray): Original discharge values.
        x_new (Optional[np.ndarray]): New time points for interpolation. Defaults to None.

    Returns:
        np.ndarray: Interpolated discharge values.
    """
```

#### apply_correction

```python
def apply_correction(
    self,
    modified_discharge: np.ndarray,
    smoothing_enabled: bool = True,
    interpolation_enabled: bool = True,
) -> np.ndarray:
    """Apply the complete correction algorithm.

    Args:
        modified_discharge (np.ndarray): Modified discharge data.
        smoothing_enabled (bool): Whether to enable five-point smoothing. Defaults to True.
        interpolation_enabled (bool): Whether to enable spline interpolation. Defaults to True.

    Returns:
        np.ndarray: Corrected discharge data.
    """
```

## Functions

### apply_smooth_correction

```python
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
        original_data (pd.DataFrame): Original data.
        modified_data (pd.DataFrame): Modified data.
        discharge_column (str): Name of discharge column. Defaults to "gen_discharge".
        time_column (str): Name of time column. Defaults to "time".

    Returns:
        pd.DataFrame: Data with smoothing correction applied.

    Raises:
        ValueError: If required columns are missing.
    """
```

### apply_water_balance_correction

```python
def apply_water_balance_correction(
    original_data: pd.DataFrame,
    modified_data: pd.DataFrame,
    discharge_column: str = "gen_discharge",
    time_column: str = "time",
) -> pd.DataFrame:
    """Apply water balance correction algorithm.

    Args:
        original_data (pd.DataFrame): Original data.
        modified_data (pd.DataFrame): Modified data.
        discharge_column (str): Name of discharge column. Defaults to "gen_discharge".
        time_column (str): Name of time column. Defaults to "time".

    Returns:
        pd.DataFrame: Data with water balance correction applied.

    Raises:
        ValueError: If required columns are missing.
    """
```

### calculate_water_balance_metrics

```python
def calculate_water_balance_metrics(
    data: pd.DataFrame,
    net_rain_column: str = "net_rain",
    discharge_column: str = "gen_discharge",
    time_step_hours: float = 1.0,
) -> dict:
    """Calculate water balance metrics.

    Args:
        data (pd.DataFrame): Input data.
        net_rain_column (str): Name of net rain column. Defaults to "net_rain".
        discharge_column (str): Name of discharge column. Defaults to "gen_discharge".
        time_step_hours (float): Time step in hours. Defaults to 1.0.

    Returns:
        dict: Dictionary containing water balance metrics:
            - total_net_rain_mm: Total net rainfall in mm
            - total_discharge_mm: Total discharge in mm
            - balance_error_percent: Water balance error percentage
            - discharge_stats: Dictionary of discharge statistics
    """
```

### validate_correction_quality

```python
def validate_correction_quality(
    original_data: pd.DataFrame,
    corrected_data: pd.DataFrame,
    discharge_column: str = "gen_discharge",
) -> dict:
    """Validate correction quality.

    Args:
        original_data (pd.DataFrame): Original data.
        corrected_data (pd.DataFrame): Corrected data.
        discharge_column (str): Name of discharge column. Defaults to "gen_discharge".

    Returns:
        dict: Dictionary containing quality metrics:
            - mse: Mean squared error
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - relative_error_percent: Relative error percentage
            - peak_preservation_ratio: Peak preservation ratio
            - original_peak: Original peak value
            - corrected_peak: Corrected peak value

    Raises:
        ValueError: If required columns are missing.
    """
```

## API Reference

::: hydroutils.hydro_correct
