"""Unit tests for SolarScope data loader."""

import pytest
import pandas as pd
from src.data.loader import (
    load_raw_generation,
    load_raw_weather,
    verify_raw_datasets,
    RawDataLoadError,
    EXPECTED_GEN_COLS,
    EXPECTED_WEATHER_COLS
)
from src.config import PLANTS


def test_verify_raw_datasets():
    results = verify_raw_datasets()
    assert len(results) == 4
    for key, exists in results.items():
        assert exists, f"Dataset file {key} should exist and be valid."


@pytest.mark.parametrize("plant_id", [4135001, 4136001])
def test_load_raw_generation_schema(plant_id):
    df = load_raw_generation(plant_id)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in EXPECTED_GEN_COLS:
        assert col in df.columns, f"Missing required column {col}"


@pytest.mark.parametrize("plant_id", [4135001, 4136001])
def test_load_raw_weather_schema(plant_id):
    df = load_raw_weather(plant_id)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in EXPECTED_WEATHER_COLS:
        assert col in df.columns, f"Missing required column {col}"


def test_invalid_plant_id():
    with pytest.raises(ValueError):
        load_raw_generation(9999999)
