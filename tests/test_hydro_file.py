import numpy as np
import json
from hydroutils.hydro_file import NumpyArrayEncoder, serialize_json_np


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
