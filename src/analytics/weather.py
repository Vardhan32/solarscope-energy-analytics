"""Weather Impact Analysis for SolarScope.

Analyzes the impact of solar irradiation, ambient temperature, module thermal heating,
and temperature derating on solar PV generation efficiency.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def analyze_weather_correlation(plant_ts_df: pd.DataFrame) -> pd.DataFrame:
    """Computes correlation matrix between weather variables and electrical output."""
    cols = [
        "AC_POWER_KW",
        "DC_POWER_KW",
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
    ]
    daylight = plant_ts_df[plant_ts_df["IS_DAYLIGHT"]][cols].dropna()
    return daylight.corr().round(3)


def calculate_thermal_derating_curve(plant_ts_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates power output and efficiency grouped by module temperature bins."""
    df = plant_ts_df[plant_ts_df["IS_DAYLIGHT"] & (plant_ts_df["IRRADIATION"] >= 0.3)].copy()

    # Temperature bins of 5 degrees
    bins = list(range(20, 75, 5))
    labels = [f"{b}-{b+5}°C" for b in bins[:-1]]
    df["TEMP_BIN"] = pd.cut(df["MODULE_TEMPERATURE"], bins=bins, labels=labels, right=False)

    derate = df.groupby("TEMP_BIN", observed=False).agg(
        SAMPLE_COUNT=("AC_POWER_KW", "count"),
        MEAN_IRRADIATION=("IRRADIATION", "mean"),
        MEAN_AC_KW=("AC_POWER_KW", "mean"),
        MEAN_EFFICIENCY_PCT=("PLANT_EFFICIENCY_PCT", "mean"),
        MEAN_PR=("INTERVAL_CAPACITY_FACTOR_PCT", "mean"),
    ).reset_index()

    # Specific generation per unit irradiance (kW / (kW/m²))
    derate["NORMALIZED_KW_PER_IRRAD"] = (
        derate["MEAN_AC_KW"] / derate["MEAN_IRRADIATION"]
    ).round(1)

    return derate.dropna(subset=["SAMPLE_COUNT"]).reset_index(drop=True)


def calculate_cell_heating_model(plant_ts_df: pd.DataFrame) -> Dict[str, float]:
    """Fits an empirical Ross / King cell temperature model:
    T_cell - T_amb = Ross_Coefficient * G (where G is in W/m^2 or kW/m^2).
    """
    daylight = plant_ts_df[plant_ts_df["IS_DAYLIGHT"] & (plant_ts_df["IRRADIATION"] >= 0.1)].copy()
    if len(daylight) < 10:
        return {"ross_coefficient_c_per_kw_m2": 30.0, "r2_score": 0.8}

    delta_t = daylight["MODULE_TEMPERATURE"] - daylight["AMBIENT_TEMPERATURE"]
    irrad = daylight["IRRADIATION"]

    # Simple slope without intercept (Delta T = k * Irrad)
    slope = (irrad * delta_t).sum() / (irrad ** 2).sum()

    # R^2
    y_pred = slope * irrad
    ss_tot = ((delta_t - delta_t.mean()) ** 2).sum()
    ss_res = ((delta_t - y_pred) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "thermal_heating_coefficient": round(float(slope), 2),
        "r2_score": round(float(r2), 3),
        "max_delta_t_deg_c": round(float(delta_t.max()), 1),
        "avg_delta_t_deg_c": round(float(delta_t.mean()), 1),
    }


def classify_sky_conditions(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Classifies monitoring days into Clear Sky, Intermittent Clouds, or Overcast."""
    df = daily_df.copy()

    # Standard peak sun hours in western India summer is ~6.0 - 7.5 kWh/m2
    conditions = []
    for _, row in df.iterrows():
        psh = row["TOTAL_IRRADIATION_KWH_M2"]
        if psh >= 6.5:
            conditions.append("Clear Sky")
        elif psh >= 4.5:
            conditions.append("Partly Cloudy")
        else:
            conditions.append("Overcast / Rainy")

    df["SKY_CONDITION"] = conditions
    return df
