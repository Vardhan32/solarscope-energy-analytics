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
