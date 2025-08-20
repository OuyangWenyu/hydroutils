"""
Author: Wenyu Ouyang
Date: 2023-10-25 20:07:14
LastEditTime: 2025-01-15 11:55:24
LastEditors: Wenyu Ouyang
Description: Use rich to log: https://rich.readthedocs.io/en/latest/
FilePath: \\hydroutils\\hydroutils\\hydro_log.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import datetime
import logging
import os
from rich.console import Console
from rich.text import Text

from hydroutils.hydro_file import get_cache_dir


class HydroWarning:
    """A class for displaying formatted warning messages using Rich console.

    This class provides methods for displaying different types of warning messages
    with consistent formatting and color coding using the Rich library.

    Attributes:
        console (Console): Rich console instance for formatted output.
    """

    def __init__(self):
        """Initialize HydroWarning with a Rich console instance."""
        self.console = Console()

    def no_directory(self, directory_name, message=None):
        """Display a warning message for a missing directory.

        Args:
            directory_name (str): Name of the missing directory.
            message (Text, optional): Custom warning message. If None, a default
                message will be created. Defaults to None.
        """
        if message is None:
            message = Text(
                f"There is no such directory: {directory_name}", style="bold red"
            )
        self.console.print(message)

    def file_not_found(self, file_name, message=None):
        """Display a warning message for a file that cannot be found.

        Args:
            file_name (str): Name of the file that could not be found.
            message (Text, optional): Custom warning message. If None, a default
                message will be created. Defaults to None.
        """
        if message is None:
            message = Text(
                f"We didn't find this file: {file_name}", style="bold yellow"
            )
        self.console.print(message)

    def operation_successful(self, operation_detail, message=None):
        """Display a success message for a completed operation.

        Args:
            operation_detail (str): Description of the successful operation.
            message (Text, optional): Custom success message. If None, a default
                message will be created. Defaults to None.
        """
        if message is None:
            message = Text(f"Operation Success: {operation_detail}", style="bold green")
        self.console.print(message)


def hydro_logger(cls):
    """Class decorator that adds a configured logger to the decorated class.

    This decorator sets up a logger with both file and console handlers. The file
    handler writes all logs (DEBUG and above) to a timestamped file in the cache
    directory, while the console handler shows INFO and above messages.

    Args:
        cls: The class to be decorated.

    Returns:
        The decorated class with an added logger attribute.

    Example:
        @hydro_logger
        class MyClass:
            def my_method(self):
                self.logger.info("This will be logged")
    """
    # Use the class name as the logger name
    logger_name = f"{cls.__module__}.{cls.__name__}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    cache_dir = get_cache_dir()
    log_dir = os.path.join(cache_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{logger_name}_{current_time}.log")
    # Check if handlers have already been added to avoid duplication
    if not logger.handlers:
        # Create a file handler to write logs to the specified file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create a console handler to output logs to the console (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # set the format of the log
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Bind the logger to the class attribute
    cls.logger = logger
    return cls
