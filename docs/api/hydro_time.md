# hydro_time

The `hydro_time` module provides utilities for handling time-related operations in hydrological data processing.

## Core Functions

### t2str

```python
def t2str(t_: Union[str, datetime.datetime]) -> Union[datetime.datetime, str]
```

Converts between datetime string and datetime object.

**Example:**
```python
import hydroutils as hu
from datetime import datetime

# String to datetime
dt = hu.t2str('2023-01-01')  # Returns datetime(2023, 1, 1)

# Datetime to string
s = hu.t2str(datetime(2023, 1, 1))  # Returns '2023-01-01'
```

### date_to_julian

```python
def date_to_julian(a_time: Union[str, datetime.datetime]) -> int
```

Converts a date to its Julian day (day of year).

### t_range_days

```python
def t_range_days(t_range: list, *, step: np.timedelta64 = np.timedelta64(1, "D")) -> np.array
```

Creates a uniformly-spaced array of dates from a date range.

**Example:**
```python
import hydroutils as hu
import numpy as np

dates = hu.t_range_days(['2000-01-01', '2000-01-05'])
# Returns array(['2000-01-01', '2000-01-02', '2000-01-03', '2000-01-04'])
```

### t_range_days_timedelta

```python
def t_range_days_timedelta(t_array: np.array, td: int = 12, td_type: str = "h") -> np.array
```

Adds a time delta to each date in an array.

### generate_start0101_time_range

```python
def generate_start0101_time_range(
    start_time: Union[str, pd.Timestamp],
    end_time: Union[str, pd.Timestamp],
    freq: str = "8D"
) -> pd.DatetimeIndex
```

Generates a time range that resets to January 1st each year.

### t_days_lst2range

```python
def t_days_lst2range(t_array: list) -> list
```

Converts a list of dates to a date range [start, end].

### assign_time_start_end

```python
def assign_time_start_end(time_ranges: list, assign_way: str = "intersection") -> tuple
```

Determines overall start and end times from multiple time ranges.

### calculate_utc_offset

```python
def calculate_utc_offset(lat: float, lng: float, date: datetime.datetime = None) -> int
```

Calculates UTC offset for a location using tzfpy.

### get_year

```python
def get_year(a_time: Union[datetime.date, np.datetime64, str]) -> int
```

Extracts the year from various date formats.

### intersect

```python
def intersect(t_lst1: np.array, t_lst2: np.array) -> tuple
```

Finds indices where two time arrays intersect.

## API Reference

::: hydroutils.hydro_time