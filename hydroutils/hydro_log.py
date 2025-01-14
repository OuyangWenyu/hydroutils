"""
Author: Wenyu Ouyang
Date: 2023-10-25 20:07:14
LastEditTime: 2025-01-14 08:47:41
LastEditors: Wenyu Ouyang
Description: Use rich to log: https://rich.readthedocs.io/en/latest/
FilePath: \hydroutils\hydroutils\hydro_log.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from rich.console import Console
from rich.text import Text


class HydroWarning:
    def __init__(self):
        self.console = Console()

    def no_directory(self, directory_name, message=None):
        if message is None:
            message = Text(
                f"There is no such directory: {directory_name}", style="bold red"
            )
        self.console.print(message)

    def file_not_found(self, file_name, message=None):
        if message is None:
            message = Text(
                f"We didn't find this file: {file_name}", style="bold yellow"
            )
        self.console.print(message)

    def operation_successful(self, operation_detail, message=None):
        if message is None:
            message = Text(f"Operation Success: {operation_detail}", style="bold green")
        self.console.print(message)
