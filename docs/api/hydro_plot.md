# hydro_plot - Visualization Tools

The `hydro_plot` module provides specialized plotting functions for visualizing hydrological data and analysis results.

## Overview

This module contains functions for:

- **Time Series Plots**: Visualize streamflow, precipitation, and other time series data
- **Statistical Plots**: Create plots for model performance assessment
- **Flow Analysis Plots**: Flow duration curves, hydrographs, and flow statistics
- **Comparison Plots**: Side-by-side comparisons of observed vs simulated data

## Quick Example

```python
import hydroutils as hu
import pandas as pd
import numpy as np

# Sample data
dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
observed = 10 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
simulated = observed * 0.95 + np.random.normal(0, 1.5, len(dates))

# Create time series plot
fig, ax = hu.plot_timeseries(
    dates, observed, simulated,
    labels=['Observed', 'Simulated'],
    title='Streamflow Comparison'
)

# Create performance scatter plot
fig, ax = hu.plot_scatter_performance(
    observed, simulated,
    add_stats=True,  # Add NSE, R², etc.
    add_1to1_line=True
)
```

## API Reference

::: hydroutils.hydro_plot