import pytest
import pandas as pd
from hydroutils.hydro_time import generate_start0101_time_range
from hydroutils.hydro_time import assign_time_start_end


def test_generate_start0101_time_range_basic():
    start_time = "2023-12-25"
    end_time = "2024-01-10"
    expected = pd.to_datetime(["2023-12-25", "2024-01-01", "2024-01-09"])
    result = generate_start0101_time_range(start_time, end_time, freq="8D")
    pd.testing.assert_index_equal(result, expected)


def test_generate_start0101_time_range_multiple_years():
    start_time = "2022-12-25"
    end_time = "2024-01-10"
    expected = pd.to_datetime(
        [
            "2022-12-25",
            "2023-01-01",
            "2023-01-09",
            "2023-01-17",
            "2023-01-25",
            "2023-02-02",
            "2023-02-10",
            "2023-02-18",
            "2023-02-26",
            "2023-03-06",
            "2023-03-14",
            "2023-03-22",
            "2023-03-30",
            "2023-04-07",
            "2023-04-15",
            "2023-04-23",
            "2023-05-01",
            "2023-05-09",
            "2023-05-17",
            "2023-05-25",
            "2023-06-02",
            "2023-06-10",
            "2023-06-18",
            "2023-06-26",
            "2023-07-04",
            "2023-07-12",
            "2023-07-20",
            "2023-07-28",
            "2023-08-05",
            "2023-08-13",
            "2023-08-21",
            "2023-08-29",
            "2023-09-06",
            "2023-09-14",
            "2023-09-22",
            "2023-09-30",
            "2023-10-08",
            "2023-10-16",
            "2023-10-24",
            "2023-11-01",
            "2023-11-09",
            "2023-11-17",
            "2023-11-25",
            "2023-12-03",
            "2023-12-11",
            "2023-12-19",
            "2023-12-27",
            "2024-01-01",
            "2024-01-09",
        ]
    )
    result = generate_start0101_time_range(start_time, end_time, freq="8D")
    pd.testing.assert_index_equal(result, expected)


def test_generate_start0101_time_range_different_freq():
    start_time = "2023-12-25"
    end_time = "2024-01-10"
    expected = pd.to_datetime(["2023-12-25", "2024-01-01", "2024-01-08"])
    result = generate_start0101_time_range(start_time, end_time, freq="7D")
    pd.testing.assert_index_equal(result, expected)


def test_generate_start0101_time_range_no_reset():
    start_time = "2023-12-25"
    end_time = "2024-01-05"
    expected = pd.to_datetime(["2023-12-25", "2024-01-01"])
    result = generate_start0101_time_range(start_time, end_time, freq="8D")
    pd.testing.assert_index_equal(result, expected)


def test_generate_start0101_time_range_single_day():
    start_time = "2023-01-01"
    end_time = "2023-01-01"
    expected = pd.to_datetime(["2023-01-01"])
    result = generate_start0101_time_range(start_time, end_time, freq="8D")
    pd.testing.assert_index_equal(result, expected)


def test_generate_start0101_time_range_invalid_dates():
    with pytest.raises(ValueError):
        generate_start0101_time_range("invalid-date", "2024-01-10", freq="8D")


def test_assign_time_start_end_intersection():
    time_ranges = [["2023-01-01", "2023-12-31"], ["2023-06-01", "2023-12-01"]]
    expected = ("2023-06-01", "2023-12-01")
    result = assign_time_start_end(time_ranges, assign_way="intersection")
    assert result == expected


def test_assign_time_start_end_union():
    time_ranges = [["2023-01-01", "2023-12-31"], ["2023-06-01", "2023-12-01"]]
    expected = ("2023-01-01", "2023-12-31")
    result = assign_time_start_end(time_ranges, assign_way="union")
    assert result == expected


def test_assign_time_start_end_single_range():
    time_ranges = [["2023-01-01", "2023-12-31"]]
    expected = ("2023-01-01", "2023-12-31")
    result = assign_time_start_end(time_ranges, assign_way="intersection")
    assert result == expected


def test_assign_time_start_end_no_overlap():
    time_ranges = [["2023-01-01", "2023-06-01"], ["2023-07-01", "2023-12-01"]]
    expected = ("2023-07-01", "2023-06-01")
    result = assign_time_start_end(time_ranges, assign_way="intersection")
    assert result == expected


def test_assign_time_start_end_invalid_assign_way():
    time_ranges = [["2023-01-01", "2023-12-31"], ["2023-06-01", "2023-12-01"]]
    with pytest.raises(NotImplementedError):
        assign_time_start_end(time_ranges, assign_way="invalid")
