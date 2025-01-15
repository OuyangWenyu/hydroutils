"""
Author: Wenyu Ouyang
Date: 2025-01-15 11:39:07
LastEditTime: 2025-01-15 12:05:43
LastEditors: Wenyu Ouyang
Description: Test the hydro_logger decorator in hydroutils/hydro_log.py
FilePath: \hydroutils\tests\test_hydro_log.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import os
import pytest
from hydroutils.hydro_log import hydro_logger
from hydroutils.hydro_file import get_cache_dir


# Mock get_cache_dir to return a temporary directory for testing
@pytest.fixture(autouse=True)
def mock_get_cache_dir(monkeypatch, tmpdir):
    monkeypatch.setattr("hydroutils.hydro_file.get_cache_dir", lambda: tmpdir)


# Dummy class to test the hydro_logger decorator
@hydro_logger
class DummyClass:
    pass


def test_hydro_logger_adds_logger_attribute():
    dummy_instance = DummyClass()
    assert hasattr(dummy_instance, "logger")
    assert isinstance(dummy_instance.logger, logging.Logger)


def test_hydro_logger_logs_to_console(caplog):
    dummy_instance = DummyClass()
    with caplog.at_level(logging.INFO):
        dummy_instance.logger.info("Test info message")
    assert "Test info message" in caplog.text
