# hydro_file

The `hydro_file` module provides utilities for file operations, including downloading, compression, serialization, and cache management.

## Download Functions

### download_zip_files

```python
def download_zip_files(urls: list, the_dir: Path) -> None
```

Downloads multiple files from multiple URLs in parallel.

**Example:**
```python
from pathlib import Path
urls = [
    'https://example.com/file1.zip',
    'https://example.com/file2.zip'
]
download_zip_files(urls, Path('./downloads'))
```

### download_one_zip

```python
def download_one_zip(data_url: str, data_dir: str) -> list
```

Downloads and extracts a zip file from a URL.

**Example:**
```python
files = download_one_zip('https://example.com/data.zip', './downloads')
print(f"Downloaded and extracted: {files}")
```

### download_small_zip

```python
def download_small_zip(data_url: str, data_dir: str) -> None
```

Downloads and extracts a small zip file using urllib.

### download_small_file

```python
def download_small_file(data_url: str, temp_file: str) -> None
```

Downloads a small text file from a URL.

### download_excel

```python
def download_excel(data_url: str, temp_file: str) -> None
```

Downloads an Excel file from a URL.

### download_a_file_from_google_drive

```python
def download_a_file_from_google_drive(drive, dir_id: str, download_dir: str) -> None
```

Downloads files and folders from Google Drive recursively.

## Compression Functions

### zip_extract

```python
def zip_extract(the_dir: Path) -> None
```

Extracts all zip files in a directory.

### unzip_file

```python
def unzip_file(data_zip: str, path_unzip: str) -> None
```

Extracts a zip file to a specified directory.

### unzip_nested_zip

```python
def unzip_nested_zip(dataset_zip: str, path_unzip: str) -> None
```

Recursively extracts a zip file and any nested zip files within it.

## JSON Functions

### serialize_json

```python
def serialize_json(my_dict: dict, my_file: str, encoding: str = "utf-8", ensure_ascii: bool = True) -> None
```

Saves a dictionary to a JSON file.

### unserialize_json

```python
def unserialize_json(my_file: str) -> dict
```

Loads a JSON file into a dictionary.

### unserialize_json_ordered

```python
def unserialize_json_ordered(my_file: str) -> OrderedDict
```

Loads a JSON file into an OrderedDict, preserving key order.

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

### get_latest_file_in_a_lst

```python
def get_latest_file_in_a_lst(lst: list) -> str
```

Gets the most recently modified file from a list of files.

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

**Example:**
```python
import numpy as np
data = {'array': np.array([1, 2, 3])}
json_str = json.dumps(data, cls=NumpyArrayEncoder)
```