# Frequently Asked Questions (FAQ)

## Installation and Setup

### Q: How do I install hydroutils?

**A:** The easiest way is using pip:
```bash
pip install hydroutils
```

For the latest development version:
```bash
pip install git+https://github.com/zhuanglaihong/hydroutils.git
```

### Q: What Python versions are supported?

**A:** hydroutils supports Python 3.8 and higher. We recommend using Python 3.10 or later for the best performance and compatibility.

### Q: I'm getting import errors. What should I do?

**A:** First, ensure all dependencies are installed:
```bash
pip install --upgrade hydroutils
```

If you're still having issues, try installing in a fresh virtual environment:
```bash
python -m venv hydroutils-env
source hydroutils-env/bin/activate  # On Windows: hydroutils-env\Scripts\activate
pip install hydroutils
```

## Usage Questions

### Q: How do I calculate basic hydrological statistics?

**A:** Use the `stat_error` function:
```python
import hydroutils as hu
import numpy as np

observed = np.array([10.5, 12.3, 8.7, 15.2, 11.8])
simulated = np.array([10.1, 12.8, 8.9, 14.7, 11.2])

stats = hu.stat_error(observed, simulated)
print(f"NSE: {stats['NSE'][0]:.3f}")
print(f"RMSE: {stats['RMSE'][0]:.3f}")
```

### Q: Can I handle missing data (NaN values)?

**A:** Yes, most functions automatically handle NaN values by excluding them from calculations:
```python
obs_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
sim_with_nan = np.array([1.1, 2.2, 3.1, 4.2, np.nan])

# This works fine - NaN values are automatically excluded
stats = hu.stat_error(obs_with_nan, sim_with_nan)
```

### Q: How do I convert between different flow units?

**A:** Use the `streamflow_unit_conv` function:
```python
# Convert from cubic meters per second to cubic feet per second
flow_cms = np.array([10.5, 12.3, 8.7])
flow_cfs = hu.streamflow_unit_conv(flow_cms, from_unit='cms', to_unit='cfs')
```

### Q: What performance metrics are available?

**A:** hydroutils provides many standard hydrological metrics:
- **NSE**: Nash-Sutcliffe Efficiency
- **KGE**: Kling-Gupta Efficiency  
- **RMSE**: Root Mean Square Error
- **Bias**: Mean Error
- **Corr**: Pearson Correlation Coefficient
- **R2**: Coefficient of Determination
- **FHV/FLV**: High/Low Flow Volume metrics

## Data Processing

### Q: How do I process multiple time series at once?

**A:** Use the `stat_errors` function for batch processing:
```python
# Multiple stations data (5 stations, 100 time steps each)
observed_series = np.random.rand(5, 100)
simulated_series = observed_series + np.random.normal(0, 0.1, (5, 100))

# Calculate statistics for all series
all_stats = hu.stat_errors(observed_series, simulated_series)

# Extract NSE values for all stations
nse_values = [stats['NSE'][0] for stats in all_stats]
```

### Q: Can I work with pandas DataFrames?

**A:** Yes, you can easily work with pandas DataFrames:
```python
import pandas as pd

# Convert DataFrame columns to numpy arrays
df = pd.read_csv('streamflow_data.csv')
obs = df['observed'].values
sim = df['simulated'].values

stats = hu.stat_error(obs, sim)
```

### Q: How do I handle different time intervals?

**A:** Use the time processing functions:
```python
# Detect time interval automatically
time_series = pd.date_range('2020-01-01', periods=100, freq='D')
interval = hu.detect_time_interval(time_series)

# Validate unit compatibility
is_compatible = hu.validate_unit_compatibility('cms', 'streamflow')
```

## Visualization

### Q: How do I create basic plots?

**A:** Use the hydro_plot module:
```python
import matplotlib.pyplot as plt

# Time series plot
fig, ax = hu.plot_timeseries(
    dates, observed, simulated,
    labels=['Observed', 'Simulated'],
    title='Streamflow Comparison'
)
plt.show()

# Performance scatter plot
fig, ax = hu.plot_scatter_performance(
    observed, simulated,
    add_stats=True,
    add_1to1_line=True
)
plt.show()
```

### Q: Can I customize plot appearance?

**A:** Yes, hydroutils provides several styling options:
```python
# Set hydrological plot style
hu.set_hydro_plot_style()

# Use hydrological color schemes
colors = hu.get_hydro_colors(data_type='streamflow')
```

## Advanced Features

### Q: How do I use AWS S3 integration?

**A:** First configure your AWS credentials, then use S3 functions:
```python
# Upload data to S3
hu.upload_to_s3(
    local_file='data.csv',
    bucket='my-hydro-data',
    s3_key='station_001/data.csv'
)

# Download from S3
hu.download_from_s3(
    bucket='my-hydro-data',
    s3_key='station_001/data.csv',
    local_file='downloaded_data.csv'
)
```

### Q: How do I enable logging for my analysis?

**A:** Use the logging utilities:
```python
# Setup logger
logger = hu.setup_hydro_logger(
    name='my_analysis',
    log_file='analysis.log',
    level='INFO'
)

# Log your analysis steps
logger.info("Starting streamflow analysis")
stats = hu.stat_error(observed, simulated)
logger.info(f"NSE calculated: {stats['NSE'][0]:.3f}")
```

## Troubleshooting

### Q: I'm getting unexpected NSE values. What could be wrong?

**A:** Check these common issues:
1. **Data alignment**: Ensure observed and simulated data have the same time periods
2. **Missing values**: Make sure missing data is properly handled
3. **Data quality**: Check for outliers or unrealistic values
4. **Array dimensions**: Verify that arrays have the same shape

```python
# Debug your data
print(f"Observed shape: {observed.shape}")
print(f"Simulated shape: {simulated.shape}")
print(f"NaN count in observed: {np.isnan(observed).sum()}")
print(f"NaN count in simulated: {np.isnan(simulated).sum()}")
```

### Q: Why am I getting poor performance metrics?

**A:** Consider these factors:
1. **Model quality**: The underlying model may need improvement
2. **Data period**: Performance can vary by season or flow conditions
3. **Metric selection**: Different metrics emphasize different aspects of performance
4. **Data preprocessing**: Check if data normalization or transformation is needed

### Q: Functions are running slowly. How can I improve performance?

**A:** Try these optimization strategies:
1. **Use appropriate data types**: Convert to float32 if high precision isn't needed
2. **Process in chunks**: For very large datasets, process data in smaller chunks
3. **Vectorize operations**: Use NumPy operations instead of loops
4. **Consider memory usage**: Monitor memory consumption for large arrays

```python
# Example of chunked processing
def process_large_dataset(large_array, chunk_size=10000):
    results = []
    for i in range(0, len(large_array), chunk_size):
        chunk = large_array[i:i+chunk_size]
        result = hu.stat_error(chunk['obs'], chunk['sim'])
        results.append(result)
    return results
```

## Getting Help

### Q: Where can I find more examples?

**A:** Check these resources:
1. **Usage Guide**: Detailed examples in the [Usage](usage.md) section
2. **API Documentation**: Complete function reference in [API Reference](api/hydroutils.md)
3. **GitHub Examples**: Example notebooks in the repository
4. **Community**: Ask questions in GitHub Issues

### Q: How do I report bugs or request features?

**A:** Please use the GitHub Issues:
1. **Bug Reports**: [Create a bug report](https://github.com/zhuanglaihong/hydroutils/issues/new?template=bug_report.md)
2. **Feature Requests**: [Request a new feature](https://github.com/zhuanglaihong/hydroutils/issues/new?template=feature_request.md)
3. **Questions**: Use the [Discussions](https://github.com/zhuanglaihong/hydroutils/discussions) section

### Q: Can I contribute to the project?

**A:** Yes! We welcome contributions. See the [Contributing Guide](contributing.md) for details on:
- Setting up a development environment
- Code style guidelines
- Testing requirements
- Submitting pull requests

### Q: Is there a citation for academic use?

**A:** Yes, if you use hydroutils in academic research, please cite:
```
@software{hydroutils,
  author = {Your Name},
  title = {hydroutils: A Python package for hydrological analysis},
  url = {https://github.com/zhuanglaihong/hydroutils},
  version = {X.X.X},
  year = {2024}
}
```

---

## Still Need Help?

If your question isn't answered here:

1. **Search existing issues**: [GitHub Issues](https://github.com/zhuanglaihong/hydroutils/issues)
2. **Ask a question**: [GitHub Discussions](https://github.com/zhuanglaihong/hydroutils/discussions)
3. **Email support**: [Contact Information]

We're here to help you succeed with your hydrological analysis!