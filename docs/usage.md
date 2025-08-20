# Usage Guide

This guide provides practical examples of how to use `hydroutils` for common hydrological analysis tasks.

## Getting Started

```python
import hydroutils as hu
import numpy as np
import pandas as pd
```

## 1. Statistical Analysis

### Basic Hydrological Statistics

Calculate common hydrological performance metrics:

```python
# Sample data
observed = np.array([10.5, 12.3, 8.7, 15.2, 11.8, 9.4, 13.6])
simulated = np.array([10.1, 12.8, 8.9, 14.7, 11.2, 9.8, 13.1])

# Calculate comprehensive statistics
stats = hu.stat_error(observed, simulated)

print(f"Nash-Sutcliffe Efficiency (NSE): {stats['NSE'][0]:.3f}")
print(f"Root Mean Square Error (RMSE): {stats['RMSE'][0]:.3f}")
print(f"Bias: {stats['Bias'][0]:.3f}")
print(f"Correlation: {stats['Corr'][0]:.3f}")
print(f"Kling-Gupta Efficiency (KGE): {stats['KGE'][0]:.3f}")
```

### Kling-Gupta Efficiency

Calculate KGE individually:

```python
kge_value = hu.KGE(simulated, observed)
print(f"KGE: {kge_value:.3f}")
```

### Flow Duration Curve Analysis

```python
# Calculate flow duration curve slope
fms_value = hu.fms(observed, simulated, lower=0.2, upper=0.7)
print(f"Flow Duration Curve Middle Slope: {fms_value:.3f}")
```

## 2. Time Period Processing

### Unit Conversions

Convert between different streamflow units:

```python
# Convert cubic meters per second to cubic feet per second
flow_cms = np.array([10.5, 12.3, 8.7, 15.2])
flow_mm3h = hu.streamflow_unit_conv(flow_cms, basin_area, source_unit='m^3/s', target_unit='mm/3h')
print(f"Flow in mm/3h: {flow_mm3h}")

# Detect time interval
time_series = pd.date_range('2020-01-01', periods=100, freq='D')
interval = hu.detect_time_interval(time_series)
print(f"Detected interval: {interval}")
```

### Time Interval Validation

```python
# Validate unit compatibility
is_compatible = hu.validate_unit_compatibility('cms', 'streamflow')
print(f"CMS compatible with streamflow: {is_compatible}")

# Get time interval information
interval_info = hu.get_time_interval_info('1D')
print(f"Daily interval info: {interval_info}")
```

## 3. Data Processing with Files

### Reading and Processing Data

```python
# Example of processing a CSV file with hydrological data
data = pd.read_csv('streamflow_data.csv', parse_dates=['date'])

# Calculate statistics for multiple stations
stations = ['station_001', 'station_002', 'station_003']
results = {}

for station in stations:
    if f'{station}_obs' in data.columns and f'{station}_sim' in data.columns:
        obs = data[f'{station}_obs'].dropna()
        sim = data[f'{station}_sim'].dropna()
        
        # Align data
        min_length = min(len(obs), len(sim))
        obs = obs[:min_length]
        sim = sim[:min_length]
        
        results[station] = hu.stat_error(obs.values, sim.values)
        
# Display results
for station, stats in results.items():
    print(f"\n{station}:")
    print(f"  NSE: {stats['NSE'][0]:.3f}")
    print(f"  RMSE: {stats['RMSE'][0]:.3f}")
```

## 4. Advanced Statistical Analysis

### Statistical Transformations

```python
# Calculate statistical properties
flow_data = np.random.lognormal(2, 1, 1000)  # Log-normal distributed flow

# Basic statistics
basic_stats = hu.cal_stat(flow_data)
print(f"Basic statistics: {basic_stats}")

# Gamma transformation statistics
gamma_stats = hu.cal_stat_gamma(flow_data)
print(f"Gamma-transformed statistics: {gamma_stats}")

# Four key statistical indices
four_stats = hu.cal_4_stat_inds(flow_data)
print(f"P10, P90, Mean, Std: {four_stats}")
```

### Empirical Cumulative Distribution Function

```python
# Calculate ECDF
sorted_data, probabilities = hu.ecdf(flow_data)

# Plot ECDF (requires matplotlib)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(sorted_data, probabilities)
plt.xlabel('Flow')
plt.ylabel('Probability')
plt.title('Empirical Cumulative Distribution Function')
plt.grid(True)
plt.show()
```

## 5. Working with Multiple Time Series

### Batch Processing

```python
# Process multiple time series
observed_series = np.random.rand(5, 100)  # 5 stations, 100 time steps
simulated_series = observed_series + np.random.normal(0, 0.1, (5, 100))

# Calculate statistics for all series
all_stats = hu.stat_errors(observed_series, simulated_series)

# Extract NSE values for all stations
nse_values = [stats['NSE'][0] for stats in all_stats]
print(f"NSE values for all stations: {nse_values}")
```

## 6. Practical Example: Complete Workflow

Here's a complete example of a typical hydrological analysis workflow:

```python
import hydroutils as hu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
def load_sample_data():
    """Generate sample hydrological data"""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    # Simulate observed streamflow with seasonal pattern
    base_flow = 10 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    observed = base_flow + np.random.normal(0, 2, len(dates))
    
    # Simulate model predictions with some bias and error
    simulated = observed * 0.95 + np.random.normal(0, 1.5, len(dates))
    
    return pd.DataFrame({
        'date': dates,
        'observed': observed,
        'simulated': simulated
    })

# 2. Load and prepare data
df = load_sample_data()
print(f"Data shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# 3. Calculate comprehensive statistics
stats = hu.stat_error(df['observed'].values, df['simulated'].values)

print("\nPerformance Metrics:")
print(f"NSE: {stats['NSE'][0]:.3f}")
print(f"KGE: {stats['KGE'][0]:.3f}")
print(f"RMSE: {stats['RMSE'][0]:.3f}")
print(f"Bias: {stats['Bias'][0]:.3f}")
print(f"Correlation: {stats['Corr'][0]:.3f}")

# 4. Additional analysis
kge_individual = hu.KGE(df['simulated'].values, df['observed'].values)
print(f"KGE (individual calculation): {kge_individual:.3f}")

# 5. Unit conversion example
basin_area = 1000  # km^2
flow_mm3h = hu.streamflow_unit_conv(df['observed'].values, basin_area, source_unit='m^3/s', target_unit='mm/3h')
print(f"Mean flow: {df['observed'].mean():.1f} m^3/s = {flow_mm3h.mean():.1f} mm/3h")

# 6. Visualization (optional)
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(df['date'], df['observed'], label='Observed', alpha=0.7)
plt.plot(df['date'], df['simulated'], label='Simulated', alpha=0.7)
plt.ylabel('Streamflow (cms)')
plt.title('Time Series Comparison')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.scatter(df['observed'], df['simulated'], alpha=0.5)
plt.plot([df['observed'].min(), df['observed'].max()], 
         [df['observed'].min(), df['observed'].max()], 'r--')
plt.xlabel('Observed (cms)')
plt.ylabel('Simulated (cms)')
plt.title(f'Scatter Plot (NSE: {stats["NSE"][0]:.3f})')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nAnalysis complete!")
```

## 7. Error Handling and Best Practices

### Handling Missing Data

```python
# Sample data with NaN values
obs_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
sim_with_nan = np.array([1.1, 2.2, 3.1, 4.2, np.nan])

# The stat_error function automatically handles NaN values
try:
    stats = hu.stat_error(obs_with_nan, sim_with_nan)
    print("Statistics calculated successfully with NaN handling")
except Exception as e:
    print(f"Error: {e}")
```

### Data Validation

```python
# Validate input data before analysis
def validate_data(observed, simulated):
    """Validate input data for hydrological analysis"""
    
    if len(observed) != len(simulated):
        raise ValueError("Observed and simulated data must have same length")
    
    if len(observed) == 0:
        raise ValueError("Data arrays cannot be empty")
    
    valid_obs = ~np.isnan(observed)
    valid_sim = ~np.isnan(simulated)
    valid_both = valid_obs & valid_sim
    
    if np.sum(valid_both) < 10:
        print("Warning: Less than 10 valid data points")
    
    return valid_both

# Example usage
obs = np.random.rand(100)
sim = obs + np.random.normal(0, 0.1, 100)

# Add some NaN values
obs[5:10] = np.nan
sim[15:20] = np.nan

valid_mask = validate_data(obs, sim)
print(f"Valid data points: {np.sum(valid_mask)}/{len(obs)}")
```

## Next Steps

- Explore the complete [API Reference](api/hydroutils.md) for all available functions
- Check out [specific module documentation](api/hydro_plot.md) for specialized features
- See [Contributing Guidelines](contributing.md) if you want to add new features
- Visit [FAQ](faq.md) for common questions and troubleshooting
