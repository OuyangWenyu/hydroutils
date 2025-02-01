"""
Author: Wenyu Ouyang
Date: 2025-02-01 21:58:43
LastEditTime: 2025-02-02 06:31:18
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydroutils\tests\test_hydro_file.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import numpy as np
import json
from hydroutils.hydro_file import NumpyArrayEncoder, serialize_json_np, serialize_json


def test_default_with_ndarray():
    encoder = NumpyArrayEncoder()
    array = np.array([1, 2, 3])
    result = encoder.default(array)
    assert result == [1, 2, 3]


def test_default_with_integer():
    encoder = NumpyArrayEncoder()
    integer = np.int32(42)
    result = encoder.default(integer)
    assert result == 42


def test_default_with_floating():
    encoder = NumpyArrayEncoder()
    floating = np.float64(3.14)
    result = encoder.default(floating)
    assert result == 3.14


def test_default_with_all_numpy():
    encoder = NumpyArrayEncoder()
    array = np.array([np.int64(1), 2, 3])
    result = encoder.default(array)
    assert result == [1, 2, 3]


def test_serialize_json_np(tmp_path):
    test_dict = {
        "array": np.array([1, 2, 3]),
        "integer": np.int32(42),
        "floating": np.float64(3.14),
    }
    test_file = tmp_path / "test.json"
    serialize_json_np(test_dict, test_file)

    with open(test_file, "r") as f:
        data = json.load(f)

    assert data["array"] == [1, 2, 3]
    assert data["integer"] == 42
    assert data["floating"] == 3.14


def test_serialize_json(tmp_path):
    test_dict = {"name": "John", "age": 30, "city": "New York"}
    test_file = tmp_path / "test_serialize.json"
    serialize_json(test_dict, test_file)

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["name"] == "John"
    assert data["age"] == 30
    assert data["city"] == "New York"


def test_serialize_json_with_non_ascii(tmp_path):
    test_dict = {"name": "张三", "age": 30, "city": "大连"}
    test_file = tmp_path / "test_serialize_non_ascii.json"
    serialize_json(test_dict, test_file, ensure_ascii=False)

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["name"] == "张三"
    assert data["age"] == 30
    assert data["city"] == "大连"
