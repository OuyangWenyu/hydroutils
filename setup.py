#!/usr/bin/env python
"""
Author: Wenyu Ouyang
Date: 2022-12-02 10:42:19
LastEditTime: 2023-07-31 17:11:11
LastEditors: Wenyu Ouyang
Description: The setup script.
FilePath: /hydroutils/setup.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


import io
import pathlib
from os import path as op
from setuptools import setup, find_packages

readme = pathlib.Path("README.md").read_text()
here = op.abspath(op.dirname(__file__))

# get the dependencies and installs
with io.open(op.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if "git+" not in x]

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Wenyu Ouyang",
    author_email="wenyuouyang@outlook.com",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="A collection of commonly used util functions in hydrological modeling",
    entry_points={
        "console_scripts": [
            "hydroutils=hydroutils.cli:main",
        ],
    },
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="hydroutils",
    name="hydroutils",
    packages=find_packages(include=["hydroutils", "hydroutils.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/OuyangWenyu/hydroutils",
    version='0.0.2',
    zip_safe=False,
)
