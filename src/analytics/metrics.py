"""Core Energy Analytics Metrics for SolarScope.

Implements daily, monthly, and diurnal generation calculations, Capacity Factor (CF),
Performance Ratio (PR per IEC 61724), Peak/Average powers, and Specific Yield.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import PLANTS, PlantMetadata, STC_IRRADIANCE


def calculate_daily_metrics(plant_ts_df: pd.DataFrame, meta: PlantMetadata) -> pd.DataFrame:
    """Calculates daily aggregated energy, peak power, capacity factor, and PR."""
    df = plant_ts_df.copy()

    # Aggregate daily
    daily = df.groupby("DATE").agg(
        TOTAL_AC_ENERGY_KWH=("INTERVAL_ENERGY_KWH", "sum"),
        TOTAL_DC_ENERGY_KWH=("DC_POWER_KW", lambda x: (x * 0.25).sum()),
        PEAK_AC_POWER_KW=("AC_POWER_KW", "max"),
        PEAK_DC_POWER_KW=("DC_POWER_KW", "max"),
        AVG_DAY_AC_POWER_KW=("AC_POWER_KW", lambda x: x[df.loc[x.index, "IS_DAYLIGHT"]].mean() if df.loc[x.index, "IS_DAYLIGHT"].any() else 0.0),
        TOTAL_IRRADIATION_KWH_M2=("IRRADIATION", lambda x: (x * 0.25).sum()),
        PEAK_MODULE_TEMP=("MODULE_TEMPERATURE", "max"),
        AVG_MODULE_TEMP=("MODULE_TEMPERATURE", "mean"),
        AVG_AMBIENT_TEMP=("AMBIENT_TEMPERATURE", "mean"),
    ).reset_index()

    daily["DATE"] = pd.to_datetime(daily["DATE"])

    # Specific Yield (kWh / kWp): Total Generation / Plant DC Capacity
    total_dc_kw = meta.inverter_count * meta.nameplate_dc_kw_per_inverter
    daily["SPECIFIC_YIELD_KWH_KWP"] = daily["TOTAL_AC_ENERGY_KWH"] / total_dc_kw

    # Capacity Factor (%): Daily kWh / (Total AC Capacity kW * 24 hours) * 100
    daily["CAPACITY_FACTOR_PCT"] = (
        (daily["TOTAL_AC_ENERGY_KWH"] / (meta.total_ac_capacity_kw * 24.0)) * 100.0
    ).clip(0.0, 100.0)

    # Performance Ratio (PR) per IEC 61724:
    # PR = Yield_final / Yield_reference = (Actual_kWh / P_dc_rated) / (Irradiation_kWh_m2 / STC_1kW_m2)
    reference_yield = daily["TOTAL_IRRADIATION_KWH_M2"] / STC_IRRADIANCE
    daily["PERFORMANCE_RATIO"] = np.where(
        reference_yield > 0.1,
        (daily["SPECIFIC_YIELD_KWH_KWP"] / reference_yield).clip(0.0, 1.2),
        np.nan
    )

    # Inverter conversion efficiency fleet daily average
    valid = daily["TOTAL_DC_ENERGY_KWH"] > 0
    daily["FLEET_EFFICIENCY_PCT"] = np.nan
    daily.loc[valid, "FLEET_EFFICIENCY_PCT"] = (
        (daily.loc[valid, "TOTAL_AC_ENERGY_KWH"] / daily.loc[valid, "TOTAL_DC_ENERGY_KWH"]) * 100.0
    ).clip(0.0, 100.0)

    # Rolling 7-day averages
    daily["ROLLING_7D_AC_KWH"] = daily["TOTAL_AC_ENERGY_KWH"].rolling(window=7, min_periods=1).mean()
    daily["ROLLING_7D_CF_PCT"] = daily["CAPACITY_FACTOR_PCT"].rolling(window=7, min_periods=1).mean()

    return daily.sort_values("DATE").reset_index(drop=True)


def calculate_diurnal_profile(plant_ts_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates hourly diurnal curves with mean, median, 10th and 90th percentiles."""
    diurnal = plant_ts_df.groupby("HOUR").agg(
        MEAN_AC_KW=("AC_POWER_KW", "mean"),
        MEDIAN_AC_KW=("AC_POWER_KW", "median"),
        P10_AC_KW=("AC_POWER_KW", lambda x: np.percentile(x, 10)),
        P90_AC_KW=("AC_POWER_KW", lambda x: np.percentile(x, 90)),
        MEAN_DC_KW=("DC_POWER_KW", "mean"),
        MEAN_IRRADIATION=("IRRADIATION", "mean"),
        MEAN_MODULE_TEMP=("MODULE_TEMPERATURE", "mean"),
        MEAN_AMBIENT_TEMP=("AMBIENT_TEMPERATURE", "mean"),
    ).reset_index()

    return diurnal


def calculate_kpi_summary(plant_ts_df: pd.DataFrame, daily_df: pd.DataFrame, meta: PlantMetadata) -> Dict[str, Any]:
    """Computes headline KPIs for dashboard executive summary cards."""
    total_energy_kwh = daily_df["TOTAL_AC_ENERGY_KWH"].sum()
    total_energy_mwh = total_energy_kwh / 1000.0
    total_energy_gwh = total_energy_mwh / 1000.0

    peak_ac_kw = plant_ts_df["AC_POWER_KW"].max()
    peak_dc_kw = plant_ts_df["DC_POWER_KW"].max()

    avg_daily_kwh = daily_df["TOTAL_AC_ENERGY_KWH"].mean()
    avg_cf_pct = daily_df["CAPACITY_FACTOR_PCT"].mean()
    avg_pr = daily_df["PERFORMANCE_RATIO"].dropna().mean()
    avg_efficiency_pct = daily_df["FLEET_EFFICIENCY_PCT"].dropna().mean()

    total_days = daily_df["DATE"].nunique()
    total_sun_hours = daily_df["TOTAL_IRRADIATION_KWH_M2"].sum()

    return {
        "plant_id": meta.plant_id,
        "plant_name": meta.name,
        "location": meta.location,
        "installed_capacity_mw": meta.total_ac_capacity_mw,
        "inverter_count": meta.inverter_count,
        "monitoring_days": int(total_days),
        "total_energy_mwh": round(total_energy_mwh, 2),
        "total_energy_gwh": round(total_energy_gwh, 4),
        "avg_daily_mwh": round(avg_daily_kwh / 1000.0, 2),
        "peak_ac_kw": round(peak_ac_kw, 1),
        "peak_dc_kw": round(peak_dc_kw, 1),
        "avg_capacity_factor_pct": round(avg_cf_pct, 2),
        "avg_performance_ratio": round(avg_pr, 3) if not np.isnan(avg_pr) else 0.85,
        "fleet_efficiency_pct": round(avg_efficiency_pct, 2),
        "cumulative_irradiance_kwh_m2": round(total_sun_hours, 1),
    }
