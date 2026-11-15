"""Unit tests for SolarScope machine learning forecasting module."""

import pytest
import pandas as pd
import numpy as np
from src.config import PLANTS
from src.models.baseline import Persistence24hModel, DiurnalClimatologyModel
from src.models.evaluation import evaluate_forecast, compare_model_metrics
from src.models.forecasting import engineer_features, chronological_split


def test_evaluate_forecast_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])  # error = 10 each
    capacity_kw = 1000.0

    metrics = evaluate_forecast(y_true, y_pred, capacity_kw, "TestModel")
    assert metrics["mae_kw"] == 10.0
    assert metrics["rmse_kw"] == 10.0
    assert metrics["nmae_pct"] == 1.0  # 10 / 1000 * 100
    assert metrics["r2_score"] > 0.99
    assert metrics["wape_pct"] == 4.0  # 40 / 1000 * 100


def test_baseline_persistence_model():
    m = Persistence24hModel(intervals_per_day=4)
    X = pd.DataFrame({"LAG_96": [10.0, 20.0, 30.0, 40.0]})
    preds = m.predict(X)
    assert np.array_equal(preds, [10.0, 20.0, 30.0, 40.0])


def test_baseline_diurnal_climatology():
    m = DiurnalClimatologyModel()
    train_X = pd.DataFrame({
        "HOUR": [12, 12, 13, 13],
        "MINUTE": [0, 0, 0, 0]
    })
    train_y = pd.Series([500.0, 600.0, 400.0, 450.0])
    m.fit(train_X, train_y)

    test_X = pd.DataFrame({
        "HOUR": [12, 13],
        "MINUTE": [0, 0]
    })
    preds = m.predict(test_X)
    assert preds[0] == 550.0
    assert preds[1] == 425.0


def test_chronological_split():
    dates = pd.date_range("2020-05-01", periods=1000, freq="15min")
    df = pd.DataFrame({
        "DATE_TIME": dates,
        "AC_POWER_KW": np.random.rand(1000)
    })
    train_df, test_df = chronological_split(df, test_days=2)
    assert len(test_df) == 2 * 96
    assert len(train_df) == 1000 - 192
    # Ensure strict temporal ordering (max train date < min test date)
    assert train_df["DATE_TIME"].max() < test_df["DATE_TIME"].min()



def test_engineered_lags_never_backfill_from_future_rows():
    dates = pd.date_range("2020-05-01", periods=220, freq="15min")
    base = pd.DataFrame({
        "DATE_TIME": dates,
        "HOUR": dates.hour,
        "MINUTE": dates.minute,
        "AC_POWER_KW": np.arange(len(dates), dtype=float),
    })
    changed_future = base.copy()
    changed_future.loc[96, "AC_POWER_KW"] = 99999.0

    base_features = engineer_features(base)
    changed_features = engineer_features(changed_future)

    # The first lag is unavailable at row 0. A later measurement must not
    # be copied backwards into that earlier row.
    assert base_features.loc[0, "LAG_96"] == 0.0
    assert changed_features.loc[0, "LAG_96"] == 0.0
    # Once a real 24-hour history exists, the lag points to the prior day.
    assert base_features.loc[96, "LAG_96"] == base.loc[0, "AC_POWER_KW"]
    assert changed_features.loc[96, "LAG_96"] == changed_future.loc[0, "AC_POWER_KW"]


def test_evaluate_forecast_edge_cases_and_clipping():
    # Negative predictions must be clipped to zero (physical solar generation >= 0)
    y_true = np.array([0.0, 50.0])
    y_pred = np.array([-20.0, 50.0])  # -20 clipped to 0 -> error on first is 0, second is 0
    metrics = evaluate_forecast(y_true, y_pred, capacity_kw=100.0, model_name="ClippedModel")
    assert metrics["mae_kw"] == 0.0

    # Empty or all-NaN arrays
    empty_metrics = evaluate_forecast(np.array([]), np.array([]), capacity_kw=100.0)
    assert empty_metrics["mae_kw"] == 0.0
    assert empty_metrics["r2_score"] == 0.0

    # Zero capacity should not raise ZeroDivisionError
    zero_cap_metrics = evaluate_forecast(np.array([10.0]), np.array([10.0]), capacity_kw=0.0)
    assert zero_cap_metrics["nmae_pct"] == 0.0
    assert zero_cap_metrics["nrmse_pct"] == 0.0


def test_compare_model_metrics_sorting():
    metrics_list = [
        {"model_name": "ModelB", "mae_kw": 25.0, "rmse_kw": 30.0, "r2_score": 0.8},
        {"model_name": "ModelA", "mae_kw": 10.0, "rmse_kw": 15.0, "r2_score": 0.95},
    ]
    df = compare_model_metrics(metrics_list)
    assert df.iloc[0]["model_name"] == "ModelA"
    assert df.iloc[1]["model_name"] == "ModelB"

