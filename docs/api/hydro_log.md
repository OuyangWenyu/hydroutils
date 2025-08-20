# hydro_log

The `hydro_log` module provides logging utilities with rich formatting and file/console output capabilities.

## Classes

### HydroWarning

A class for handling and displaying hydrology-related warnings and messages with rich formatting.

```python
class HydroWarning:
    def __init__(self)
    def no_directory(directory_name: str, message: Text = None) -> None
    def file_not_found(file_name: str, message: Text = None) -> None
    def operation_successful(operation_detail: str, message: Text = None) -> None
```

**Example:**
```python
from hydroutils import HydroWarning

warning = HydroWarning()

# Display directory not found warning
warning.no_directory("/path/to/missing/dir")

# Display file not found warning
warning.file_not_found("data.csv")

# Display success message
warning.operation_successful("Data processing complete")
```

## Decorators

### @hydro_logger

A class decorator that adds logging capabilities to a class.

**Example:**
```python
from hydroutils import hydro_logger

@hydro_logger
class MyHydroClass:
    def process_data(self):
        self.logger.info("Starting data processing...")
        # Processing logic here
        self.logger.debug("Processing complete")

# The class now has both file and console logging
obj = MyHydroClass()
obj.process_data()  # Logs will be written to file and console
```

**Features:**
- Automatically creates log directory in cache
- Timestamps in log filenames
- Both file (DEBUG level) and console (INFO level) output
- Standard logging format with timestamp, module name, and log level