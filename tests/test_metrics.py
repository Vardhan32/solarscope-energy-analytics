"""Unit tests for SolarScope metrics and analytics calculations."""

import pytest
import pandas as pd
import numpy as np
from src.config import PLANTS
from src.analytics.metrics import (
    calculate_daily_metrics,
    calculate_diurnal_profile,
    calculate_kpi_summary,
)


@pytest.fixture
def sample_plant_ts():
    meta = PLANTS[4135001]
    dates = pd.date_range("2020-05-15 00:00:00", "2020-05-16 23:45:00", freq="15min")
    n = len(dates)

    # Synthetic diurnal bell curve
    hours = dates.hour + dates.minute / 60.0
    irrad = np.where((hours >= 6) & (hours <= 18), np.sin(np.pi * (hours - 6) / 12) * 0.9, 0.0)
    ac_kw = irrad * meta.total_ac_capacity_kw * 0.8
    dc_kw = ac_kw / 0.98

    df = pd.DataFrame({
        "DATE_TIME": dates,
        "DATE": dates.date,
        "HOUR": dates.hour,
        "MINUTE": dates.minute,
        "DAY_OF_WEEK": dates.day_of_week,
        "DAY_OF_YEAR": dates.day_of_year,
        "AC_POWER_KW": ac_kw,
        "DC_POWER_KW": dc_kw,
        "EXPECTED_AC_POWER_KW": ac_kw * 1.02,
        "INTERVAL_ENERGY_KWH": ac_kw * 0.25,
        "IRRADIATION": irrad,
        "AMBIENT_TEMPERATURE": 25.0 + irrad * 10,
        "MODULE_TEMPERATURE": 28.0 + irrad * 25,
        "IS_DAYLIGHT": irrad > 0.05,
        "PLANT_EFFICIENCY_PCT": np.where(dc_kw > 10, 98.0, np.nan),
        "INTERVAL_CAPACITY_FACTOR_PCT": (ac_kw / meta.total_ac_capacity_kw) * 100.0,
    })
    return df


def test_calculate_daily_metrics(sample_plant_ts):
    meta = PLANTS[4135001]
    daily = calculate_daily_metrics(sample_plant_ts, meta)

    assert len(daily) == 2
    assert "TOTAL_AC_ENERGY_KWH" in daily.columns
    assert "CAPACITY_FACTOR_PCT" in daily.columns
    assert "PERFORMANCE_RATIO" in daily.columns
    assert 0.0 <= daily["CAPACITY_FACTOR_PCT"].iloc[0] <= 100.0
    assert 0.5 <= daily["PERFORMANCE_RATIO"].iloc[0] <= 1.2


def test_calculate_diurnal_profile(sample_plant_ts):
    diurnal = calculate_diurnal_profile(sample_plant_ts)
    assert len(diurnal) == 24
    assert "MEAN_AC_KW" in diurnal.columns
    assert "MEDIAN_AC_KW" in diurnal.columns
    assert "P10_AC_KW" in diurnal.columns
    assert "P90_AC_KW" in diurnal.columns
    # Night hours should be 0
    assert diurnal.loc[diurnal["HOUR"] == 1, "MEAN_AC_KW"].iloc[0] == 0.0
    # Noon hour should be > 0
    assert diurnal.loc[diurnal["HOUR"] == 12, "MEAN_AC_KW"].iloc[0] > 0.0


def test_calculate_kpi_summary(sample_plant_ts):
    meta = PLANTS[4135001]
    daily = calculate_daily_metrics(sample_plant_ts, meta)
    kpi = calculate_kpi_summary(sample_plant_ts, daily, meta)

    assert kpi["plant_id"] == 4135001
    assert kpi["installed_capacity_mw"] == meta.total_ac_capacity_mw
    assert kpi["total_energy_mwh"] > 0
    assert kpi["avg_capacity_factor_pct"] > 0
