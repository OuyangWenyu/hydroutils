# hydro_file

The `hydro_file` module provides utilities for file I/O operations, including reading and writing various data formats commonly used in hydrological applications.

## Core Functions

### read_ts_xrdataset

```python
def read_ts_xrdataset(
    file_path: str,
    var_name: str = None,
    time_name: str = "time",
    lat_name: str = "lat",
    lon_name: str = "lon"
) -> xr.Dataset
```

Reads time series data from NetCDF files into xarray Dataset format.

**Example:**
```python
import hydroutils as hu

# Read NetCDF file
ds = hu.read_ts_xrdataset('data.nc', var_name='precipitation')
print(f"Dataset shape: {ds.dims}")
```

### write_ts_xrdataset

```python
def write_ts_xrdataset(
    ds: xr.Dataset,
    file_path: str,
    var_name: str = None,
    encoding: dict = None
) -> None
```

Writes xarray Dataset to NetCDF file.

### read_csv

```python
def read_csv(file_path: str, **kwargs) -> pd.DataFrame
```

Reads CSV files with enhanced error handling and encoding detection.

### write_csv

```python
def write_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None
```

Writes DataFrame to CSV with proper encoding and error handling.

## JSON Functions

### serialize_json

```python
def serialize_json(my_dict: dict, my_file: str) -> None
```

Saves a dictionary to a JSON file.

### unserialize_json

```python
def unserialize_json(my_file: str) -> dict
```

Loads a JSON file into a dictionary.

### serialize_json_np

```python
def serialize_json_np(my_dict: dict, my_file: str) -> None
```

Saves a dictionary containing NumPy arrays to a JSON file.

## Pickle Functions

### serialize_pickle

```python
def serialize_pickle(my_object: object, my_file: str) -> None
```

Saves an object to a pickle file.

### unserialize_pickle

```python
def unserialize_pickle(my_file: str) -> object
```

Loads an object from a pickle file.

## NumPy Array Functions

### serialize_numpy

```python
def serialize_numpy(my_array: np.ndarray, my_file: str) -> None
```

Saves a NumPy array to a .npy file.

### unserialize_numpy

```python
def unserialize_numpy(my_file: str) -> np.ndarray
```

Loads a NumPy array from a .npy file.

## File Management Functions

### get_lastest_file_in_a_dir

```python
def get_lastest_file_in_a_dir(dir_path: str) -> str
```

Gets the most recently modified .pth file in a directory.

### get_cache_dir

```python
def get_cache_dir(app_name: str = "hydro") -> str
```

Gets the appropriate cache directory for the current platform.

## Classes

### NumpyArrayEncoder

```python
class NumpyArrayEncoder(json.JSONEncoder)
```

JSON encoder that handles NumPy arrays and scalars.

## API Reference

::: hydroutils.hydro_file