# hydro_stat

The `hydro_stat` module provides statistical functions for hydrological data analysis, including error metrics, flow duration curves, and data transformations.

## Error Metrics

### stat_error

```python
def stat_error(target: np.ndarray, pred: np.ndarray, fill_nan: str = "no") -> dict
```

Calculates multiple error metrics between predicted and target values.

**Example:**
```python
obs = np.array([[1, 2, 3], [4, 5, 6]])  # 2 basins, 3 timesteps
pred = np.array([[1.1, 2.1, 3.1], [4.2, 5.1, 5.9]])
metrics = stat_error(obs, pred)
print(f"Mean RMSE: {np.mean(metrics['RMSE']):.2f}")
```

### stat_errors

```python
def stat_errors(target: np.ndarray, pred: np.ndarray, fill_nan: list = None) -> list
```

Similar to stat_error but handles 3D arrays for multiple variables.

### KGE

```python
def KGE(xs: np.ndarray, xo: np.ndarray) -> float
```

Calculates Kling-Gupta Efficiency between simulated and observed values.

## Flow Duration Curves

### cal_fdc

```python
def cal_fdc(data: np.ndarray, quantile_num: int = 100) -> np.ndarray
```

Calculates flow duration curves for multiple time series.

**Example:**
```python
flows = np.random.lognormal(0, 1, (2, 365))  # 2 locations, 365 days
fdcs = cal_fdc(flows, quantile_num=100)
```

### fms

```python
def fms(obs: np.ndarray, sim: np.ndarray, lower: float = 0.2, upper: float = 0.7) -> float
```

Calculates the slope of the middle section of the flow duration curve.

## Peak Analysis

### mean_peak_timing

```python
def mean_peak_timing(
    obs: np.ndarray,
    sim: np.ndarray,
    window: int = None,
    resolution: str = "1D",
    datetime_coord: str = None
) -> float
```

Calculates mean difference in peak flow timing between observed and simulated flows.

## Statistical Tests

### wilcoxon_t_test

```python
def wilcoxon_t_test(xs: np.ndarray, xo: np.ndarray) -> tuple
```

Performs Wilcoxon signed-rank test on paired samples.

### wilcoxon_t_test_for_lst

```python
def wilcoxon_t_test_for_lst(x_lst: list, rnd_num: int = 2) -> tuple
```

Performs pairwise Wilcoxon tests between all arrays in a list.

## Data Transformations

### cal_stat_gamma

```python
def cal_stat_gamma(x: np.ndarray) -> list
```

Transforms data to approximate normal distribution and calculates statistics.

### cal_stat_prcp_norm

```python
def cal_stat_prcp_norm(x: np.ndarray, meanprep: np.ndarray) -> list
```

Normalizes data by mean precipitation and calculates statistics.

### trans_norm

```python
def trans_norm(
    x: np.ndarray,
    var_lst: Union[str, list],
    stat_dict: dict,
    *,
    to_norm: bool
) -> np.ndarray
```

Normalizes or denormalizes data using pre-computed statistics.

## Basic Statistics

### cal_4_stat_inds

```python
def cal_4_stat_inds(b: np.ndarray) -> list
```

Calculates four basic statistical indices for an array.

### cal_stat

```python
def cal_stat(x: np.ndarray) -> list
```

Calculates basic statistics for an array, ignoring NaN values.

## Data Processing

### remove_abnormal_data

```python
def remove_abnormal_data(
    data: np.ndarray,
    *,
    q1: float = 0.00001,
    q2: float = 0.99999
) -> np.ndarray
```

Removes extreme values from data using quantile thresholds.

### month_stat_for_daily_df

```python
def month_stat_for_daily_df(df: pd.DataFrame) -> pd.DataFrame
```

Calculates monthly statistics from daily data.

## Distribution Functions

### ecdf

```python
def ecdf(data: np.ndarray) -> tuple
```

Computes empirical cumulative distribution function (ECDF).