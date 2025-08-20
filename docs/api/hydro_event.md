# hydro_event

The `hydro_event` module provides utilities for flood event extraction and analysis from hydrological time series data.

## Core Functions

### extract_flood_events

```python
def extract_flood_events(
    df: pd.DataFrame,
    warmup_length: int = 0,
    flood_event_col: str = "flood_event",
    time_col: str = "time"
) -> List[Dict]
```

Extracts flood events from a DataFrame based on a binary flood event indicator.

**Args:**
- `df`: DataFrame with flood_event and time columns
- `warmup_length`: Number of time steps to include as warmup period
- `flood_event_col`: Name of flood event indicator column
- `time_col`: Name of time column

**Returns:**
- List of flood event dictionaries containing event data and metadata

**Example:**
```python
import hydroutils as hu
import pandas as pd

# Sample data with flood events
df = pd.DataFrame({
    'time': pd.date_range('2020-01-01', periods=10),
    'flood_event': [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    'flow': [100, 120, 200, 300, 250, 150, 130, 180, 220, 140]
})

# Extract flood events
events = hu.extract_flood_events(df, warmup_length=1)
print(f"Found {len(events)} flood events")
```

### time_to_ten_digits

```python
def time_to_ten_digits(time_obj) -> str
```

Converts time objects to ten-digit format (YYYYMMDDHH).

**Example:**
```python
from datetime import datetime
import hydroutils as hu

dt = datetime(2020, 1, 1, 12, 0)
time_str = hu.time_to_ten_digits(dt)  # Returns '2020010112'
```

### extract_peaks

```python
def extract_peaks(
    data: np.ndarray,
    threshold: float = None,
    min_distance: int = 1
) -> Tuple[np.ndarray, np.ndarray]
```

Extracts peak values and their indices from time series data.

### calculate_event_statistics

```python
def calculate_event_statistics(events: List[Dict]) -> pd.DataFrame
```

Calculates statistical summary for extracted flood events.

## API Reference

::: hydroutils.hydro_event
