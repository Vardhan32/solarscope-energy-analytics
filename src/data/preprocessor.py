"""Data Preprocessing, Cleaning, and Feature Engineering Module for SolarScope."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd

from src.config import (
    PLANTS,
    PlantMetadata,
    STC_IRRADIANCE,
    STC_MODULE_TEMP,
    TEMP_COEFFICIENT_PMAX,
    DAYLIGHT_MIN_IRRADIANCE,
)
from src.data.loader import load_raw_generation, load_raw_weather
from src.utils.logger import get_logger

logger = get_logger("preprocessor")


def clean_generation_data(df: pd.DataFrame, meta: PlantMetadata) -> pd.DataFrame:
    """Cleans, validates, and parses datetime for inverter generation data."""
    df = df.copy()

    # Parse timestamps explicitly according to plant format
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format=meta.date_format_gen)

    # Deduplicate
    initial_len = len(df)
    df = df.drop_duplicates(subset=["DATE_TIME", "SOURCE_KEY"]).reset_index(drop=True)
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} duplicate generation rows for plant {meta.plant_id}")

    # Standardize DC Power to kW
    # Plant 1 raw DC is scaled by 10 relative to kW; Plant 2 is in kW
    df["DC_POWER_RAW"] = df["DC_POWER"]
    df["DC_POWER_KW"] = df["DC_POWER"] / meta.dc_scale_factor
    df["AC_POWER_KW"] = df["AC_POWER"]

    # Physical boundaries: generation cannot be negative
    df["DC_POWER_KW"] = df["DC_POWER_KW"].clip(lower=0.0)
    df["AC_POWER_KW"] = df["AC_POWER_KW"].clip(lower=0.0)

    # Energy in kWh generated during this 15-minute interval (Power kW * 0.25 h)
    df["INTERVAL_ENERGY_KWH"] = df["AC_POWER_KW"] * 0.25

    return df


def clean_weather_data(df: pd.DataFrame, meta: PlantMetadata) -> pd.DataFrame:
    """Cleans and validates weather sensor records."""
    df = df.copy()

    # Parse timestamps
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format=meta.date_format_weather)

    # Deduplicate weather readings on DATE_TIME
    initial_len = len(df)
    df = df.drop_duplicates(subset=["DATE_TIME"]).reset_index(drop=True)
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} duplicate weather rows for plant {meta.plant_id}")

    # Physical validity checks
    df["IRRADIATION"] = df["IRRADIATION"].clip(lower=0.0, upper=1.5)
    df["AMBIENT_TEMPERATURE"] = df["AMBIENT_TEMPERATURE"].clip(lower=-10.0, upper=60.0)
    df["MODULE_TEMPERATURE"] = df["MODULE_TEMPERATURE"].clip(lower=-10.0, upper=85.0)

    # Forward fill small gaps if present
    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    df["IRRADIATION"] = df["IRRADIATION"].ffill().bfill()
    df["AMBIENT_TEMPERATURE"] = df["AMBIENT_TEMPERATURE"].ffill().bfill()
    df["MODULE_TEMPERATURE"] = df["MODULE_TEMPERATURE"].ffill().bfill()

    return df


def merge_and_enrich(gen_df: pd.DataFrame, weather_df: pd.DataFrame, meta: PlantMetadata) -> pd.DataFrame:
    """Merges generation and weather sensors and computes analytical engineering features."""
    # Merge on DATE_TIME and PLANT_ID
    merged = pd.merge(
        gen_df,
        weather_df[[
            "DATE_TIME",
            "PLANT_ID",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION"
        ]],
        on=["DATE_TIME", "PLANT_ID"],
        how="inner"
    )

    # Sort deterministically
    merged = merged.sort_values(["DATE_TIME", "SOURCE_KEY"]).reset_index(drop=True)

    # Daylight status
    merged["IS_DAYLIGHT"] = merged["IRRADIATION"] >= DAYLIGHT_MIN_IRRADIANCE

    # Inverter conversion efficiency: AC Power / DC Power (defined when DC > 10 kW)
    valid_dc = merged["DC_POWER_KW"] > 10.0
    merged["INVERTER_EFFICIENCY_PCT"] = np.nan
    merged.loc[valid_dc, "INVERTER_EFFICIENCY_PCT"] = (
        (merged.loc[valid_dc, "AC_POWER_KW"] / merged.loc[valid_dc, "DC_POWER_KW"]) * 100.0
    ).clip(0.0, 100.0)

    # Theoretical Expected AC Generation (PVWatts / STC temperature-derated formula)
    # P_exp = P_rated * (G / G_stc) * [1 + gamma * (T_cell - 25)]
    rated_kw = meta.nameplate_ac_kw_per_inverter
    temp_loss_factor = 1.0 + TEMP_COEFFICIENT_PMAX * (merged["MODULE_TEMPERATURE"] - STC_MODULE_TEMP)
    temp_loss_factor = temp_loss_factor.clip(lower=0.5, upper=1.1)

    expected_ac = rated_kw * (merged["IRRADIATION"] / STC_IRRADIANCE) * temp_loss_factor
    merged["EXPECTED_AC_POWER_KW"] = expected_ac.clip(lower=0.0)

    # Performance Ratio (actual AC vs expected AC during daylight)
    merged["PERFORMANCE_RATIO"] = np.nan
    sun_mask = (merged["IRRADIATION"] >= 0.20) & (merged["EXPECTED_AC_POWER_KW"] > 10.0)
    merged.loc[sun_mask, "PERFORMANCE_RATIO"] = (
        merged.loc[sun_mask, "AC_POWER_KW"] / merged.loc[sun_mask, "EXPECTED_AC_POWER_KW"]
    ).clip(0.0, 1.5)

    # Diurnal and temporal calendar features
    dt = merged["DATE_TIME"]
    merged["DATE"] = dt.dt.date
    merged["HOUR"] = dt.dt.hour
    merged["MINUTE"] = dt.dt.minute
    merged["DAY_OF_WEEK"] = dt.dt.day_of_week
    merged["DAY_OF_YEAR"] = dt.dt.day_of_year

    # Cyclic time encoding
    merged["HOUR_SIN"] = np.sin(2 * np.pi * (merged["HOUR"] + merged["MINUTE"] / 60.0) / 24.0)
    merged["HOUR_COS"] = np.cos(2 * np.pi * (merged["HOUR"] + merged["MINUTE"] / 60.0) / 24.0)

    return merged


def create_plant_level_timeseries(merged_df: pd.DataFrame, meta: PlantMetadata) -> pd.DataFrame:
    """Aggregates inverter data to plant-level 15-minute timeseries."""
    agg_funcs = {
        "DC_POWER_KW": "sum",
        "AC_POWER_KW": "sum",
        "EXPECTED_AC_POWER_KW": "sum",
        "INTERVAL_ENERGY_KWH": "sum",
        "IRRADIATION": "mean",
        "AMBIENT_TEMPERATURE": "mean",
        "MODULE_TEMPERATURE": "mean",
        "IS_DAYLIGHT": "max",
    }

    plant_ts = merged_df.groupby("DATE_TIME").agg(agg_funcs).reset_index()
    plant_ts["PLANT_ID"] = meta.plant_id
    plant_ts["PLANT_NAME"] = meta.name

    # Fleet-wide inverter efficiency
    valid = plant_ts["DC_POWER_KW"] > 50.0
    plant_ts["PLANT_EFFICIENCY_PCT"] = np.nan
    plant_ts.loc[valid, "PLANT_EFFICIENCY_PCT"] = (
        (plant_ts.loc[valid, "AC_POWER_KW"] / plant_ts.loc[valid, "DC_POWER_KW"]) * 100.0
    ).clip(0.0, 100.0)

    # Capacity Factor for 15-minute interval: AC_POWER_KW / Total Capacity KW
    plant_ts["INTERVAL_CAPACITY_FACTOR_PCT"] = (
        (plant_ts["AC_POWER_KW"] / meta.total_ac_capacity_kw) * 100.0
    ).clip(0.0, 100.0)

    # Date and calendar features
    dt = plant_ts["DATE_TIME"]
    plant_ts["DATE"] = dt.dt.date
    plant_ts["HOUR"] = dt.dt.hour
    plant_ts["MINUTE"] = dt.dt.minute
    plant_ts["DAY_OF_WEEK"] = dt.dt.day_of_week
    plant_ts["DAY_OF_YEAR"] = dt.dt.day_of_year
    plant_ts["HOUR_SIN"] = np.sin(2 * np.pi * (plant_ts["HOUR"] + dt.dt.minute / 60.0) / 24.0)
    plant_ts["HOUR_COS"] = np.cos(2 * np.pi * (plant_ts["HOUR"] + dt.dt.minute / 60.0) / 24.0)

    return plant_ts.sort_values("DATE_TIME").reset_index(drop=True)


def process_plant(plant_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, cleans, enriches, and returns (inverter_level_df, plant_level_df)."""
    meta = PLANTS[plant_id]
    gen_raw = load_raw_generation(plant_id)
    weather_raw = load_raw_weather(plant_id)

    gen_clean = clean_generation_data(gen_raw, meta)
    weather_clean = clean_weather_data(weather_raw, meta)

    inverter_df = merge_and_enrich(gen_clean, weather_clean, meta)
    plant_ts_df = create_plant_level_timeseries(inverter_df, meta)

    return inverter_df, plant_ts_df
