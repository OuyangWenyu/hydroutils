# Makefile for hydroutils
.PHONY: help install install-dev test test-cov lint format clean build

# Default target
help:
	@echo "Available targets:"
	@echo "  install-dev - Install package with development dependencies"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting (ruff)"
	@echo "  format      - Format code (black + ruff)"
	@echo "  clean       - Clean build artifacts"
	@echo "  build       - Build package"
	@echo "  bump-patch  - Bump patch version"

# Installation
install-dev:
	uv sync --all-extras --dev

# Testing
test:
	uv run pytest

test-cov:
	uv run pytest --cov=hydroutils --cov-report=html --cov-report=term-missing

# Code quality
lint:
	uv run ruff check .

format:
	uv run black .
	uv run ruff format .
	uv run ruff check --fix .

# Cleaning
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Building
build: clean
	uv run python -m build

# Version bump
bump-patch:
	uv run bump2version patch

# Development setup
setup-dev: install-dev
	uv run pre-commit install
	@echo "Development environment setup complete!"