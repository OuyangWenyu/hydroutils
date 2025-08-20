# Welcome to hydroutils

[![image](https://img.shields.io/pypi/v/hydroutils.svg)](https://pypi.python.org/pypi/hydroutils)
[![image](https://pyup.io/repos/github/zhuanglaihong/hydroutils/shield.svg)](https://pyup.io/repos/github/zhuanglaihong/hydroutils)

**A collection of commonly used utility functions for hydrological modeling and analysis**

`hydroutils` is a comprehensive Python package that provides essential tools and utilities for hydrological data processing, statistical analysis, and modeling. It is designed to streamline common tasks in hydrology research and engineering applications.

## Key Features

### 📊 Statistical Analysis
- Comprehensive hydrological statistics (NSE, KGE, RMSE, Bias, etc.)
- Flow duration curve analysis
- Peak flow analysis and timing metrics
- Flood event extraction and characterization

### 🕐 Time Series Processing
- Time unit conversions and standardization
- Time interval detection and validation
- Temporal data manipulation utilities

### 📈 Data Visualization
- Specialized plotting functions for hydrological data
- Flow duration curves
- Time series plots with hydrological context

### 📁 File Operations
- NetCDF file handling
- CSV and text file processing
- Data import/export utilities

### ☁️ Cloud Integration
- AWS S3 integration for large dataset handling
- Cloud-based data storage and retrieval

### 🧮 Mathematical Operations
- Hydrological unit conversions
- Mathematical utilities for water resources calculations
- Array operations optimized for hydrological data

## Quick Start

```python
import hydroutils as hu

# Calculate hydrological statistics
nse = hu.stat_error(observed, simulated)['NSE']

# Extract flood events
events = hu.extract_flood_events(dataframe)

# Convert streamflow units
converted = hu.streamflow_unit_conv(data, from_unit='cms', to_unit='cfs')
```

## Installation

```bash
pip install hydroutils
```

## Documentation Structure

- **[Installation Guide](installation.md)** - Detailed installation instructions
- **[Usage Examples](usage.md)** - Practical examples and tutorials  
- **[API Reference](api/hydroutils.md)** - Complete API documentation
- **[Contributing](contributing.md)** - How to contribute to the project
- **[FAQ](faq.md)** - Frequently asked questions

## License & Credits

- Free software: MIT license
- Documentation: <https://zhuanglaihong.github.io/hydroutils>
- Created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template
