"""Unit tests for SolarScope preprocessor."""

import pytest
import pandas as pd
import numpy as np
from src.config import PLANTS
from src.data.preprocessor import (
    clean_generation_data,
    clean_weather_data,
    merge_and_enrich,
    create_plant_level_timeseries,
    process_plant,
)


def test_clean_generation_data():
    meta = PLANTS[4135001]
    raw_df = pd.DataFrame({
        "DATE_TIME": ["15-05-2020 00:00", "15-05-2020 00:15"],
        "PLANT_ID": [4135001, 4135001],
        "SOURCE_KEY": ["INV_01", "INV_01"],
        "DC_POWER": [1000.0, -50.0],  # negative should be clipped to 0
        "AC_POWER": [100.0, 0.0],
        "DAILY_YIELD": [0.0, 0.0],
        "TOTAL_YIELD": [10000.0, 10000.0]
    })

    cleaned = clean_generation_data(raw_df, meta)
    assert len(cleaned) == 2
    assert pd.api.types.is_datetime64_any_dtype(cleaned["DATE_TIME"])
    assert cleaned["DC_POWER_KW"].iloc[0] == 100.0  # 1000 / 10
    assert cleaned["DC_POWER_KW"].iloc[1] == 0.0    # clipped
    assert cleaned["INTERVAL_ENERGY_KWH"].iloc[0] == 25.0  # 100 * 0.25


def test_clean_weather_data():
    meta = PLANTS[4135001]
    raw_weather = pd.DataFrame({
        "DATE_TIME": ["2020-05-15 00:00:00", "2020-05-15 00:15:00"],
        "PLANT_ID": [4135001, 4135001],
        "SOURCE_KEY": ["SENSOR_01", "SENSOR_01"],
        "AMBIENT_TEMPERATURE": [25.0, 75.0],  # 75 should be clipped to 60
        "MODULE_TEMPERATURE": [28.0, 95.0],   # 95 should be clipped to 85
        "IRRADIATION": [0.0, -0.1],          # -0.1 should be clipped to 0.0
    })

    cleaned = clean_weather_data(raw_weather, meta)
    assert cleaned["AMBIENT_TEMPERATURE"].iloc[1] == 60.0
    assert cleaned["MODULE_TEMPERATURE"].iloc[1] == 85.0
    assert cleaned["IRRADIATION"].iloc[1] == 0.0


def test_process_plant():
    inverter_df, plant_ts_df = process_plant(4135001)
    assert len(inverter_df) > 50000
    assert len(plant_ts_df) > 2000
    assert "EXPECTED_AC_POWER_KW" in inverter_df.columns
    assert "INTERVAL_CAPACITY_FACTOR_PCT" in plant_ts_df.columns
    assert "MINUTE" in plant_ts_df.columns
    assert "HOUR" in plant_ts_df.columns
