import numpy as np
import pytest

from hydroutils.hydro_stat import (
    KGE,
    mae,
    nse,
    pbias,
    pearson_r,
    rmse,
    stat_error,
    stat_error_i,
    stat_errors,
)


def _oracle(obs, pred):
    """Independent paired-finite metric oracle for stat_error's no-fill path."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    result = dict.fromkeys(
        ("Bias", "RMSE", "ubRMSE", "Corr", "R2", "NSE", "KGE", "FHV", "FLV"),
        np.nan,
    )
    if obs.size == 0:
        return result

    error = pred - obs
    result["Bias"] = np.mean(error)
    result["RMSE"] = np.sqrt(np.mean(error**2))
    result["ubRMSE"] = np.sqrt(
        np.mean(((pred - np.mean(pred)) - (obs - np.mean(obs))) ** 2)
    )
    if obs.size >= 2 and np.std(obs) > 0:
        result["NSE"] = 1.0 - np.sum(error**2) / np.sum((obs - np.mean(obs)) ** 2)
        result["R2"] = result["NSE"]
    if obs.size >= 2 and np.std(obs) > 0 and np.std(pred) > 0:
        result["Corr"] = np.corrcoef(obs, pred)[0, 1]
        if np.mean(obs) != 0:
            result["KGE"] = 1.0 - np.sqrt(
                (result["Corr"] - 1.0) ** 2
                + (np.std(pred) / np.std(obs) - 1.0) ** 2
                + (np.mean(pred) / np.mean(obs) - 1.0) ** 2
            )

    obs_sorted = np.sort(obs)
    pred_sorted = np.sort(pred)
    low_end = round(0.3 * len(obs))
    high_start = round(0.98 * len(obs))
    obs_low, pred_low = obs_sorted[:low_end], pred_sorted[:low_end]
    obs_high, pred_high = obs_sorted[high_start:], pred_sorted[high_start:]
    if obs_low.size and np.sum(obs_low) != 0:
        result["FLV"] = np.sum(pred_low - obs_low) / np.sum(obs_low) * 100.0
    if obs_high.size and np.sum(obs_high) != 0:
        result["FHV"] = np.sum(pred_high - obs_high) / np.sum(obs_high) * 100.0
    return result


@pytest.mark.parametrize(
    ("obs", "pred"),
    [
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([1, np.nan, 3, 4], [1, 100, 3, 4]),
        ([1, 2, 3, 4], [1, np.nan, 3, 4]),
        ([1, np.inf, 3, 4], [1, 100, 3, 4]),
        ([1, 2, 3, 4], [1, np.inf, 3, 4]),
        ([1, np.nan, 3, 4, 5], [1, 2, np.nan, 4, 5]),
    ],
    ids=[
        "finite",
        "nan-observation",
        "nan-prediction",
        "inf-observation",
        "inf-prediction",
        "different-gaps",
    ],
)
def test_stat_error_matches_independent_common_finite_pair_oracle(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    expected = _oracle(obs, pred)
    actual = stat_error(obs[None, :], pred[None, :], fill_nan="no")
    for metric, value in expected.items():
        if np.isnan(value):
            assert np.isnan(actual[metric][0]), metric
        else:
            assert actual[metric][0] == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_original_ubrmse_bug_probe_is_fixed_and_qc_is_explicit():
    obs = np.array([[1.0, np.nan, 2.0]])
    pred = np.array([[1.0, 777.0, 2.0]])

    metrics = stat_error(obs, pred, fill_nan="no")
    metrics_with_qc, qc = stat_error(obs, pred, fill_nan="no", return_qc=True)

    assert metrics["ubRMSE"][0] == 0.0
    assert metrics_with_qc["ubRMSE"][0] == 0.0
    assert qc["total_pairs"].tolist() == [3]
    assert qc["valid_pairs"].tolist() == [2]
    assert qc["missing_observation_pairs"].tolist() == [1]
    assert qc["nonfinite_prediction_pairs"].tolist() == [0]
    assert qc["valid_fraction"].tolist() == pytest.approx([2 / 3])
    assert set(metrics) == {
        "Bias",
        "RMSE",
        "ubRMSE",
        "Corr",
        "R2",
        "NSE",
        "KGE",
        "FHV",
        "FLV",
    }


def test_no_valid_pairs_return_nan_and_one_pair_keeps_defined_metrics():
    none = stat_error(
        np.array([[np.nan, np.inf]]), np.array([[1.0, 2.0]]), return_qc=True
    )
    metrics, qc = none
    assert all(np.isnan(values[0]) for values in metrics.values())
    assert qc["valid_pairs"].tolist() == [0]
    assert qc["valid_fraction"].tolist() == [0.0]

    one = stat_error(np.array([[2.0, np.nan]]), np.array([[3.0, 4.0]]))
    assert one["Bias"][0] == 1.0
    assert one["RMSE"][0] == 1.0
    assert one["ubRMSE"][0] == 0.0
    for metric in ("Corr", "R2", "NSE", "KGE", "FHV", "FLV"):
        assert np.isnan(one[metric][0]), metric
    assert all(np.isnan(value) for value in stat_error_i([], []).values())


def test_constant_and_zero_mean_cases_are_reported_as_undefined():
    constant_obs = stat_error(np.array([[3.0, 3.0, 3.0]]), np.array([[3.0, 2.0, 3.0]]))
    for metric in ("Corr", "R2", "NSE", "KGE"):
        assert np.isnan(constant_obs[metric][0]), metric

    constant_pred = stat_error(np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 1.0, 1.0]]))
    assert np.isnan(constant_pred["Corr"][0])
    assert np.isnan(constant_pred["KGE"][0])
    assert np.isfinite(constant_pred["NSE"][0])

    assert np.isnan(KGE(np.array([1.0, 2.0]), np.array([-1.0, 1.0])))


def test_public_scalar_metric_helpers_use_paired_finite_values():
    obs = np.array([1.0, np.nan, 3.0, np.inf])
    pred = np.array([1.0, 100.0, 2.0, 7.0])
    assert mae(obs, pred) == pytest.approx(0.5)
    assert rmse(obs, pred) == pytest.approx(np.sqrt(0.5))
    # Preserve HydroErr's existing positional convention: its first argument
    # is the simulated series and its second argument is the reference series.
    assert nse(obs, pred) == pytest.approx(-1.0)
    assert pearson_r(obs, pred) == pytest.approx(1.0)
    assert np.isnan(pbias(np.array([1.0, -1.0]), np.array([2.0, 0.0])))


def test_scaled_metrics_remain_finite_when_squared_values_would_overflow():
    metrics = stat_error(
        np.array([[0.0, 0.0, 0.0]]), np.array([[1e200, -1e200, 1e200]])
    )
    assert metrics["RMSE"][0] == pytest.approx(1e200, rel=1e-12)
    assert np.isfinite(metrics["ubRMSE"][0])


def test_multibasin_multitarget_qc_preserves_variable_separation():
    obs = np.array(
        [
            [[1.0, 2.0], [np.nan, 3.0], [4.0, np.inf]],
            [[np.inf, 2.0], [5.0, np.nan], [6.0, 7.0]],
        ]
    )
    pred = np.array(
        [
            [[1.0, 2.0], [8.0, 3.0], [4.0, 5.0]],
            [[1.0, 2.0], [5.0, np.inf], [6.0, 7.0]],
        ]
    )
    metrics, qc_by_target = stat_errors(
        obs, pred, fill_nan=["no", "no"], return_qc=True
    )
    assert len(metrics) == len(qc_by_target) == 2
    assert qc_by_target[0]["valid_pairs"].tolist() == [2, 2]
    assert qc_by_target[1]["valid_pairs"].tolist() == [2, 2]
    assert qc_by_target[1]["nonfinite_prediction_pairs"].tolist() == [0, 1]
    assert metrics[0]["RMSE"][0] == 0.0
    assert metrics[1]["RMSE"][1] == 0.0

    default_metrics = stat_errors(obs, pred)
    assert len(default_metrics) == 2
    assert default_metrics[0]["RMSE"][0] == 0.0


def test_sum_and_mean_fill_modes_keep_their_historical_aggregation():
    obs = np.array([[1.0, np.nan, 2.0]])
    pred = np.array([[1.0, 777.0, 2.0]])
    summed = stat_error(obs, pred, fill_nan="sum")
    averaged = stat_error(obs, pred, fill_nan="mean")
    assert summed["Bias"][0] == pytest.approx(388.5)
    assert averaged["Bias"][0] == pytest.approx(194.0)
