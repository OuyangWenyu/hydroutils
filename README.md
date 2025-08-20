# hydroutils

[![image](https://img.shields.io/pypi/v/hydroutils.svg)](https://pypi.python.org/pypi/hydroutils)
[![image](https://img.shields.io/conda/vn/conda-forge/hydroutils.svg)](https://anaconda.org/conda-forge/hydroutils)
[![image](https://pyup.io/repos/github/OuyangWenyu/hydroutils/shield.svg)](https://pyup.io/repos/github/OuyangWenyu/hydroutils)
[![Python Version](https://img.shields.io/pypi/pyversions/hydroutils.svg)](https://pypi.org/project/hydroutils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive collection of utility functions for hydrological modeling and analysis**

Hydroutils is a Python package designed for hydrological modeling workflows, providing statistical analysis, data visualization, file handling, time period operations and unit conversion, specifically tailored for hydrological research and applications.

**This package is still under development, and the API is subject to change.**

- **Free software**: MIT license
- **Documentation**: https://OuyangWenyu.github.io/hydroutils
- **Source Code**: https://github.com/OuyangWenyu/hydroutils
- **PyPI Package**: https://pypi.org/project/hydroutils/

## ✨ Features

### 📊 Statistical Analysis (`hydro_stat`)
- **Dynamic Metric Functions**: Automatically generated statistical functions (NSE, RMSE, MAE, etc.) 
- **Multi-dimensional Analysis**: Support for 2D/3D arrays for basin-scale analysis
- **HydroErr Integration**: Standardized hydrological metrics through HydroErr package
- **NaN Handling**: Flexible strategies ('no', 'sum', 'mean') for missing data
- **Runtime Metric Addition**: Add custom metrics dynamically with `add_metric()`

### 📈 Visualization (`hydro_plot`)
- **Geospatial Plotting**: Cartopy integration for map-based visualizations
- **Chinese Font Support**: Automatic font configuration for Chinese text rendering
- **Statistical Plots**: ECDF, box plots, heatmaps, correlation matrices
- **Hydrological Specializations**: Flow duration curves, unit hydrographs, precipitation plots
- **Customizable Styling**: Extensive configuration options for colors, styles, and formats

### 📁 File Operations (`hydro_file`)
- **JSON Serialization**: NumPy array support with `NumpyArrayEncoder`
- **Cloud Storage**: S3 and MinIO integration for remote data access
- **ZIP Handling**: Nested ZIP file extraction and management
- **Cache Management**: Automatic cache directory creation and management
- **Async Operations**: Asynchronous data retrieval capabilities

### ⏰ Time Period (`hydro_time`)
- **UTC Calculations**: Timezone offset computation from coordinates
- **Date Parsing**: Flexible date string parsing and manipulation
- **Time Range Operations**: Intersection, generation, and validation
- **Interval Detection**: Automatic time interval identification

### 🏷️ Unit Conversion (`hydro_units`)
- **Streamflow Units**: Comprehensive unit conversion for hydrological variables
- **Time Interval Detection**: Automatic detection and validation of time intervals
- **Unit Compatibility**: Validation functions for unit consistency
- **Pint Integration**: Physical units handling with pint and pint-xarray

### 🌊 Event Analysis (`hydro_event`)
- **Hydrological Event Detection**: Flood event identification
- **Event Characterization**: Duration, magnitude, and timing analysis

### ☁️ Cloud Integration (`hydro_s3`)
- **AWS S3 Support**: Direct integration with Amazon S3 services
- **MinIO Compatibility**: Local and private cloud storage solutions
- **Credential Management**: Secure credential handling and configuration

### 📝 Logging (`hydro_log`)
- **Rich Console Output**: Colored and formatted console logging
- **Progress Tracking**: Advanced progress bars and status indicators
- **Debug Support**: Comprehensive debugging and error reporting

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install hydroutils

# Install with development dependencies using uv (recommended)
pip install uv
uv add hydroutils

# For development setup
git clone https://github.com/OuyangWenyu/hydroutils.git
cd hydroutils
uv sync --all-extras --dev
```

### Basic Usage

```python
import hydroutils
import numpy as np

# Statistical Analysis
obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
sim = np.array([1.1, 2.1, 2.9, 3.9, 5.1])

# Calculate Nash-Sutcliffe Efficiency
nse_value = hydroutils.nse(obs, sim)
print(f"NSE: {nse_value:.3f}")

# Multiple metrics at once
metrics = hydroutils.stat_error(obs, sim)
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")

# Visualization
import matplotlib.pyplot as plt
fig, ax = hydroutils.plot_ecdf([obs, sim], 
                               labels=['Observed', 'Simulated'],
                               colors=['blue', 'red'])
plt.show()
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/OuyangWenyu/hydroutils.git
cd hydroutils

# Install UV (modern Python package manager)
pip install uv

# Setup development environment
uv sync --all-extras --dev
```

### Development Commands

```bash
# Run tests
uv run pytest                    # Basic test run
uv run pytest --cov=hydroutils   # With coverage
make test-cov                    # With HTML coverage report

# Code formatting and linting
uv run black .                   # Format code
uv run ruff check .              # Lint code
uv run ruff check --fix .        # Fix linting issues
make format                      # Format and lint together

# Type checking
uv run mypy hydroutils
make type-check

# Documentation
uv run mkdocs serve              # Serve docs locally
make docs-serve

# Build and release
uv run python -m build           # Build package
make build
```

### Project Structure

```
hydroutils/
├── hydroutils/
│   ├── __init__.py              # Package initialization and exports
│   ├── hydro_event.py           # Hydrological event analysis
│   ├── hydro_file.py            # File I/O and cloud storage
│   ├── hydro_log.py             # Logging and console output
│   ├── hydro_plot.py            # Visualization functions
│   ├── hydro_s3.py              # AWS S3 and MinIO integration
│   ├── hydro_stat.py            # Statistical analysis engine
│   ├── hydro_time.py            # Time series utilities
│   └── hydro_units.py           # Unit conversion and validation
├── tests/                       # Comprehensive test suite
├── docs/                        # MkDocs documentation
├── pyproject.toml               # Modern Python project config
├── Makefile                     # Development convenience commands
└── uv.lock                      # UV package manager lock file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make check-all`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📖 Documentation

Comprehensive documentation is available at [https://OuyangWenyu.github.io/hydroutils](https://OuyangWenyu.github.io/hydroutils), including:

- **API Reference**: Complete function and class documentation
- **User Guide**: Step-by-step tutorials and examples
- **Contributing Guide**: Development setup and contribution guidelines
- **FAQ**: Frequently asked questions and troubleshooting

## 🏗️ Requirements

- **Python**: >=3.10
- **Core Dependencies**: numpy, pandas, matplotlib, seaborn
- **Scientific Computing**: scipy, HydroErr, numba
- **Visualization**: cartopy (for geospatial plots)
- **Cloud Storage**: boto3, minio, s3fs
- **Utilities**: tqdm, rich, xarray, pint

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HydroErr**: For standardized hydrological error metrics
- **Cookiecutter**: Project template from [giswqs/pypackage](https://github.com/giswqs/pypackage)
- **Scientific Python Ecosystem**: NumPy, SciPy, Matplotlib, Pandas

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/OuyangWenyu/hydroutils/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OuyangWenyu/hydroutils/discussions)
- **Email**: wenyuouyang@outlook.com

---

**Made with ❤️ for the hydrology community**