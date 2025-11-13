# Installation

## Requirements

`hydroutils` requires Python 3.8 or higher. The package has been tested on:

- Python 3.8, 3.9, 3.10, 3.11
- Windows, macOS, and Linux

### Core Dependencies

The following packages are automatically installed:

- `numpy` - Array operations and mathematical functions
- `pandas` - Data manipulation and analysis
- `scipy` - Scientific computing
- `matplotlib` - Plotting and visualization
- `xarray` - Labeled multi-dimensional arrays
- `netCDF4` - NetCDF file handling
- `HydroErr` - Hydrological error metrics

### Optional Dependencies

For extended functionality:

- `jupyter` - For notebook examples

## Installation Methods

### 1. Stable Release (Recommended)

Install the latest stable release from PyPI:

```bash
pip install hydroutils
```

This is the preferred method as it installs the most recent stable release with all dependencies.

### 2. Development Version

For the latest features and bug fixes, install from GitHub:

```bash
pip install git+https://github.com/zhuanglaihong/hydroutils.git
```

### 3. From Source

If you want to contribute or modify the code:

```bash
# Clone the repository
git clone https://github.com/zhuanglaihong/hydroutils.git
cd hydroutils

# Install in development mode
pip install -e .
```

### 4. With Optional Dependencies

To install with all optional dependencies:

```bash
pip install hydroutils[all]
```

Or install specific optional dependencies:

```bash
pip install hydroutils[viz]     # For advanced visualization
pip install hydroutils[dev]     # For development tools
```

## Virtual Environment Setup

We recommend using a virtual environment to avoid dependency conflicts:

### Using conda

```bash
# Create a new environment
conda create -n hydroutils python=3.10
conda activate hydroutils

# Install hydroutils
pip install hydroutils
```

### Using venv

```bash
# Create a new environment
python -m venv hydroutils-env

# Activate the environment
# On Windows:
hydroutils-env\Scripts\activate
# On macOS/Linux:
source hydroutils-env/bin/activate

# Install hydroutils
pip install hydroutils
```

## Verification

Test your installation:

```python
import hydroutils as hu
print(hu.__version__)

# Quick functionality test
import numpy as np
obs = np.random.rand(100)
sim = obs + np.random.normal(0, 0.1, 100)
stats = hu.stat_error(obs, sim)
print(f"NSE: {stats['NSE'][0]:.3f}")
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed:
   ```bash
   pip install --upgrade hydroutils
   ```

2. **Permission Denied**: Use `--user` flag:
   ```bash
   pip install --user hydroutils
   ```

3. **SSL Certificate Error**: Try with trusted hosts:
   ```bash
   pip install --trusted-host pypi.org --trusted-host pypi.python.org hydroutils
   ```

### Platform-Specific Notes

**Windows Users:**
- Consider using Anaconda for easier scientific package management
- Some dependencies may require Visual C++ Build Tools

**macOS Users:**
- Xcode command line tools may be required for some dependencies
- Use Homebrew to install system-level dependencies if needed

**Linux Users:**
- Install system dependencies for scientific packages:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-dev gfortran libopenblas-dev
  
  # CentOS/RHEL
  sudo yum install python3-devel gcc-gfortran openblas-devel
  ```

## Getting Help

If you encounter any installation issues:

1. Check the [FAQ](faq.md) for common solutions
2. Search [existing issues](https://github.com/zhuanglaihong/hydroutils/issues)
3. Create a [new issue](https://github.com/zhuanglaihong/hydroutils/issues/new) with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the problem
