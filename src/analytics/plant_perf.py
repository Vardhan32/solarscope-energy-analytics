"""Solar Plant Performance and Inverter Benchmarking Module.

Provides comparative plant performance, inverter-to-inverter benchmarking,
conversion efficiency distribution, and actual vs expected power tracking.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.config import PLANTS, PlantMetadata


def benchmark_inverters(inverter_df: pd.DataFrame, meta: PlantMetadata) -> pd.DataFrame:
    """Computes operational and health metrics for each inverter in the fleet."""
    # Filter daylight hours for efficiency & PR stats
    daylight_df = inverter_df[inverter_df["IS_DAYLIGHT"]].copy()

    grouped = inverter_df.groupby("SOURCE_KEY")

    # Inverter totals and maximums
    bench = grouped.agg(
        TOTAL_ENERGY_KWH=("INTERVAL_ENERGY_KWH", "sum"),
        PEAK_AC_POWER_KW=("AC_POWER_KW", "max"),
        PEAK_DC_POWER_KW=("DC_POWER_KW", "max"),
        AVG_DAY_AC_KW=("AC_POWER_KW", lambda x: x[inverter_df.loc[x.index, "IS_DAYLIGHT"]].mean() if inverter_df.loc[x.index, "IS_DAYLIGHT"].any() else 0.0),
        TOTAL_OPERATING_INTERVALS=("AC_POWER_KW", lambda x: (x > 1.0).sum()),
    ).reset_index()

    # Efficiency and PR from daylight readings
    day_grouped = daylight_df.groupby("SOURCE_KEY")
    eff_pr = day_grouped.agg(
        MEDIAN_EFFICIENCY_PCT=("INVERTER_EFFICIENCY_PCT", "median"),
        MEAN_EFFICIENCY_PCT=("INVERTER_EFFICIENCY_PCT", "mean"),
        MEDIAN_PR=("PERFORMANCE_RATIO", "median"),
        MEAN_PR=("PERFORMANCE_RATIO", "mean"),
    ).reset_index()

    bench = pd.merge(bench, eff_pr, on="SOURCE_KEY", how="left")

    # Specific Yield per Inverter (kWh / kWp)
    bench["SPECIFIC_YIELD_KWH_KWP"] = bench["TOTAL_ENERGY_KWH"] / meta.nameplate_dc_kw_per_inverter

    # Fleet Median comparison (Relative Performance Index)
    fleet_median_energy = bench["TOTAL_ENERGY_KWH"].median()
    bench["RELATIVE_ENERGY_INDEX"] = (bench["TOTAL_ENERGY_KWH"] / fleet_median_energy).round(3)

    # Inverter Health Status Category
    bench["HEALTH_STATUS"] = "Normal"
    bench.loc[bench["RELATIVE_ENERGY_INDEX"] < 0.90, "HEALTH_STATUS"] = "Underperforming"
    bench.loc[bench["RELATIVE_ENERGY_INDEX"] < 0.75, "HEALTH_STATUS"] = "Critical Underperformance"
    bench.loc[bench["RELATIVE_ENERGY_INDEX"] >= 1.02, "HEALTH_STATUS"] = "High Performer"

    # Sort descending by total energy
    return bench.sort_values("TOTAL_ENERGY_KWH", ascending=False).reset_index(drop=True)


def calculate_actual_vs_expected(inverter_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates actual vs theoretical expected AC power by timestamp."""
    grouped = inverter_df.groupby("DATE_TIME").agg(
        ACTUAL_AC_KW=("AC_POWER_KW", "sum"),
        EXPECTED_AC_KW=("EXPECTED_AC_POWER_KW", "sum"),
        IRRADIATION=("IRRADIATION", "mean"),
        MODULE_TEMPERATURE=("MODULE_TEMPERATURE", "mean"),
        IS_DAYLIGHT=("IS_DAYLIGHT", "max"),
    ).reset_index()

    # Filter to daylight
    grouped = grouped[grouped["IS_DAYLIGHT"]].copy()

    grouped["POWER_DEVIATION_KW"] = grouped["ACTUAL_AC_KW"] - grouped["EXPECTED_AC_KW"]
    grouped["PERCENT_DEVIATION"] = np.where(
        grouped["EXPECTED_AC_KW"] > 50.0,
        ((grouped["ACTUAL_AC_KW"] - grouped["EXPECTED_AC_KW"]) / grouped["EXPECTED_AC_KW"]) * 100.0,
        0.0
    )

    return grouped.sort_values("DATE_TIME").reset_index(drop=True)


def compare_plants(plant1_daily: pd.DataFrame, plant2_daily: pd.DataFrame) -> pd.DataFrame:
    """Creates a side-by-side comparative table between Plant 1 and Plant 2."""
    meta1 = PLANTS[4135001]
    meta2 = PLANTS[4136001]

    metrics = [
        ("Installed AC Capacity (MW)", meta1.total_ac_capacity_mw, meta2.total_ac_capacity_mw),
        ("Inverter Units", meta1.inverter_count, meta2.inverter_count),
        ("Total Generation (MWh)", plant1_daily["TOTAL_AC_ENERGY_KWH"].sum() / 1000.0, plant2_daily["TOTAL_AC_ENERGY_KWH"].sum() / 1000.0),
        ("Avg Daily Generation (MWh)", plant1_daily["TOTAL_AC_ENERGY_KWH"].mean() / 1000.0, plant2_daily["TOTAL_AC_ENERGY_KWH"].mean() / 1000.0),
        ("Peak AC Power (kW)", plant1_daily["PEAK_AC_POWER_KW"].max(), plant2_daily["PEAK_AC_POWER_KW"].max()),
        ("Avg Capacity Factor (%)", plant1_daily["CAPACITY_FACTOR_PCT"].mean(), plant2_daily["CAPACITY_FACTOR_PCT"].mean()),
        ("Avg Performance Ratio (PR)", plant1_daily["PERFORMANCE_RATIO"].dropna().mean(), plant2_daily["PERFORMANCE_RATIO"].dropna().mean()),
        ("Fleet Conversion Efficiency (%)", plant1_daily["FLEET_EFFICIENCY_PCT"].dropna().mean(), plant2_daily["FLEET_EFFICIENCY_PCT"].dropna().mean()),
        ("Specific Yield (kWh/kWp)", plant1_daily["SPECIFIC_YIELD_KWH_KWP"].sum(), plant2_daily["SPECIFIC_YIELD_KWH_KWP"].sum()),
    ]

    df = pd.DataFrame(metrics, columns=["Metric", "Plant 1 (Gandhinagar)", "Plant 2 (Gandhinagar)"])
    df["Plant 1 (Gandhinagar)"] = df["Plant 1 (Gandhinagar)"].round(2)
    df["Plant 2 (Gandhinagar)"] = df["Plant 2 (Gandhinagar)"].round(2)
    return df
