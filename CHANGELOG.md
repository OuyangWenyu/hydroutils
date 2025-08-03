# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Migration to modern Python tooling with uv and pyproject.toml
- Dynamic metric functions generation system for hydro_stat module
- Comprehensive pytest test suite for hydro_stat module
- Pre-commit hooks for code quality
- Automated CI/CD with GitHub Actions using uv
- Type hints and mypy support
- Security scanning with bandit and safety
- Modern documentation with MkDocs

### Changed
- Migrated from setup.py to pyproject.toml
- Replaced pip with uv for dependency management
- Updated GitHub Actions workflows for modern Python development
- Improved code formatting with black and ruff
- Enhanced testing with pytest and coverage reporting

### Deprecated
- setup.py (replaced by pyproject.toml)
- requirements.txt files (replaced by pyproject.toml dependencies)

### Removed
- Legacy setup.py configuration
- Old GitHub Actions workflows

### Fixed
- Code style consistency across the project
- Dependency version conflicts

### Security
- Added security scanning to CI/CD pipeline
- Implemented trusted publishing for PyPI

## [0.0.14] - 2025-01-06

### Added
- Unit hydro plot functionality
- Chinese language support in JSON serialization
- Recursive NumPy JSON serialization

### Changed
- Updated environment configuration
- Version bump to 0.0.14

### Fixed
- JSON serialization for NumPy arrays
- Chinese character handling in JSON files

## [0.0.13] - Previous Release

### Added
- Basic hydro statistics functionality
- File handling utilities
- S3 integration support
- Plotting capabilities
- Time series utilities

### Initial Features
- HydroErr integration for statistical metrics
- AWS S3 and MinIO support
- Matplotlib and Seaborn plotting
- Cartopy for GIS plotting
- Rich console output
- Async data retrieval
- Time zone handling with tzfpy

---

## Migration Notes

### From setup.py to pyproject.toml

If you're upgrading from a previous version, note the following changes:

1. **Installation**: The package can now be installed with modern tools:
   ```bash
   # With uv (recommended)
   uv add hydroutils
   
   # With pip (still supported)
   pip install hydroutils
   ```

2. **Development**: Set up development environment with:
   ```bash
   # Clone the repository
   git clone https://github.com/OuyangWenyu/hydroutils.git
   cd hydroutils
   
   # Set up development environment
   make setup-dev
   ```

3. **Testing**: Run tests with:
   ```bash
   make test
   # or
   uv run pytest
   ```

4. **Code Quality**: Format and lint code with:
   ```bash
   make format
   make lint
   ```

### Breaking Changes

- None in this release. The API remains backward compatible.

### New Features

- **Dynamic Metric Functions**: You can now use hydro statistics functions directly:
  ```python
  from hydroutils.hydro_stat import nse, rmse, mae, bias
  
  # Calculate metrics directly without importing HydroErr
  nse_value = nse(observed, simulated)
  rmse_value = rmse(observed, simulated)
  ```

- **Easy Metric Addition**: Add new metrics dynamically:
  ```python
  from hydroutils.hydro_stat import add_metric
  add_metric('new_metric', 'he_function_name', 'Description')
  ```