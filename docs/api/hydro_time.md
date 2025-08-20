# hydro_time

The `hydro_time` module provides utilities for handling time-related operations in hydrological data processing.

## Date Conversion Functions

### t2str

```python
def t2str(t_: Union[str, datetime.datetime]) -> Union[datetime.datetime, str]
```

Converts between datetime string and datetime object.

**Example:**
```python
# String to datetime
dt = t2str('2023-01-01')  # Returns datetime(2023, 1, 1)

# Datetime to string
s = t2str(datetime(2023, 1, 1))  # Returns '2023-01-01'
```

### date_to_julian

```python
def date_to_julian(a_time: Union[str, datetime.datetime]) -> int
```

Converts a date to its Julian day (day of year).

**Example:**
```python
day = date_to_julian('2023-02-01')  # Returns 32
```

## Date Range Functions

### t_range_days

```python
def t_range_days(t_range: list, *, step: np.timedelta64 = np.timedelta64(1, "D")) -> np.array
```

Creates a uniformly-spaced array of dates from a date range.

**Example:**
```python
dates = t_range_days(['2000-01-01', '2000-01-05'])
# Returns array(['2000-01-01', '2000-01-02', '2000-01-03', '2000-01-04'])
```

### t_range_days_timedelta

```python
def t_range_days_timedelta(t_array: np.array, td: int = 12, td_type: str = "h") -> np.array
```

Adds a time delta to each date in an array.

**Example:**
```python
dates = t_range_days(['2000-01-01', '2000-01-03'])
shifted = t_range_days_timedelta(dates, td=12, td_type='h')
```

### generate_start0101_time_range

```python
def generate_start0101_time_range(
    start_time: Union[str, pd.Timestamp],
    end_time: Union[str, pd.Timestamp],
    freq: str = "8D"
) -> pd.DatetimeIndex
```

Generates a time range that resets to January 1st each year.

**Example:**
```python
# Generate 8-day intervals, resetting to Jan 1st each year
dates = generate_start0101_time_range('2023-01-15', '2024-02-01', freq='8D')
```

## Time Range Operations

### t_days_lst2range

```python
def t_days_lst2range(t_array: list) -> list
```

Converts a list of dates to a date range [start, end].

**Example:**
```python
dates = ['2000-01-01', '2000-01-02', '2000-01-03', '2000-01-04']
range = t_days_lst2range(dates)  # Returns ['2000-01-01', '2000-01-04']
```

### assign_time_start_end

```python
def assign_time_start_end(time_ranges: list, assign_way: str = "intersection") -> tuple
```

Determines overall start and end times from multiple time ranges.

**Example:**
```python
ranges = [
    ['2023-01-01', '2023-12-31'],
    ['2023-03-01', '2024-02-28']
]

# Get intersection (common period)
start, end = assign_time_start_end(ranges, 'intersection')
# Returns ('2023-03-01', '2023-12-31')
```

## Timezone Functions

### calculate_utc_offset

```python
def calculate_utc_offset(lat: float, lng: float, date: datetime.datetime = None) -> int
```

Calculates UTC offset for a location using tzfpy.

**Example:**
```python
# Get current UTC offset for New York
offset = calculate_utc_offset(40.7128, -74.0060)
print(f"New York is UTC{offset:+d}")  # e.g., "UTC-4"
```

## Utility Functions

### get_year

```python
def get_year(a_time: Union[datetime.date, np.datetime64, str]) -> int
```

Extracts the year from various date formats.

**Example:**
```python
year = get_year('2023-01-01')  # Returns 2023
```

### intersect

```python
def intersect(t_lst1: np.array, t_lst2: np.array) -> tuple
```

Finds indices where two time arrays intersect.

**Example:**
```python
t1 = np.array(['2000-01-01', '2000-01-02', '2000-01-03'])
t2 = np.array(['2000-01-02', '2000-01-03', '2000-01-04'])
idx1, idx2 = intersect(t1, t2)
# idx1 = [1, 2] (indices in t1)
# idx2 = [0, 1] (indices in t2)
```