"""Solar Plant Anomaly and Underperformance Detection Module.

Identifies equipment outages, partial string failure, inverter clipping,
thermal stress, and fleet dispersion anomalies.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd

from src.config import (
    OUTAGE_IRRADIANCE_THRESHOLD,
    OUTAGE_MAX_AC_POWER,
    UNDERPERFORMANCE_RATIO,
    CLIPPING_THRESHOLD_RATIO,
    PlantMetadata,
)


def detect_operational_anomalies(inverter_df: pd.DataFrame, meta: PlantMetadata) -> pd.DataFrame:
    """Scans all inverter 15-minute intervals and returns tagged anomalies."""
    df = inverter_df.copy()

    # Calculate fleet median AC power per timestamp to use as dynamic ground-truth benchmark
    fleet_medians = df.groupby("DATE_TIME")["AC_POWER_KW"].median().rename("FLEET_MEDIAN_AC_KW")
    df = df.merge(fleet_medians, on="DATE_TIME", how="left")

    anomalies: List[Dict[str, Any]] = []

    # 1. Complete Outage / Inverter Trip during High Daylight
    outage_mask = (
        (df["IRRADIATION"] >= OUTAGE_IRRADIANCE_THRESHOLD) &
        (df["AC_POWER_KW"] <= OUTAGE_MAX_AC_POWER)
    )
    for _, row in df[outage_mask].iterrows():
        anomalies.append({
            "DATE_TIME": row["DATE_TIME"],
            "DATE": row["DATE"],
            "PLANT_ID": row["PLANT_ID"],
            "SOURCE_KEY": row["SOURCE_KEY"],
            "ANOMALY_TYPE": "Inverter Outage / Trip",
            "SEVERITY": "Critical",
            "AC_POWER_KW": round(row["AC_POWER_KW"], 2),
            "EXPECTED_AC_POWER_KW": round(row["EXPECTED_AC_POWER_KW"], 2),
            "FLEET_MEDIAN_KW": round(row["FLEET_MEDIAN_AC_KW"], 2),
            "IRRADIATION": round(row["IRRADIATION"], 3),
            "DESCRIPTION": f"Inverter produced {row['AC_POWER_KW']:.1f} kW during strong sun ({row['IRRADIATION']:.2f} kW/m²). Expected ~{row['EXPECTED_AC_POWER_KW']:.1f} kW.",
        })

    # 2. Severe Underperformance (e.g., string loss, heavy soiling, degradation)
    underperf_mask = (
        (df["IRRADIATION"] >= 0.35) &
        (df["FLEET_MEDIAN_AC_KW"] >= 200.0) &
        (df["AC_POWER_KW"] < (df["FLEET_MEDIAN_AC_KW"] * UNDERPERFORMANCE_RATIO)) &
        (df["AC_POWER_KW"] > OUTAGE_MAX_AC_POWER)  # exclude outages captured above
    )
    for _, row in df[underperf_mask].iterrows():
        anomalies.append({
            "DATE_TIME": row["DATE_TIME"],
            "DATE": row["DATE"],
            "PLANT_ID": row["PLANT_ID"],
            "SOURCE_KEY": row["SOURCE_KEY"],
            "ANOMALY_TYPE": "Severe Underperformance",
            "SEVERITY": "Warning",
            "AC_POWER_KW": round(row["AC_POWER_KW"], 2),
            "EXPECTED_AC_POWER_KW": round(row["EXPECTED_AC_POWER_KW"], 2),
            "FLEET_MEDIAN_KW": round(row["FLEET_MEDIAN_AC_KW"], 2),
            "IRRADIATION": round(row["IRRADIATION"], 3),
            "DESCRIPTION": f"Output {row['AC_POWER_KW']:.1f} kW is {((row['AC_POWER_KW']/row['FLEET_MEDIAN_AC_KW'])*100):.1f}% of fleet median ({row['FLEET_MEDIAN_AC_KW']:.1f} kW). Possible string failure or soiling.",
        })

    # 3. Inverter Power Limiting / Clipping
    clipping_limit = meta.nameplate_ac_kw_per_inverter * CLIPPING_THRESHOLD_RATIO
    clipping_mask = (df["AC_POWER_KW"] >= clipping_limit)
    for _, row in df[clipping_mask].iterrows():
        anomalies.append({
            "DATE_TIME": row["DATE_TIME"],
            "DATE": row["DATE"],
            "PLANT_ID": row["PLANT_ID"],
            "SOURCE_KEY": row["SOURCE_KEY"],
            "ANOMALY_TYPE": "Inverter Power Saturation / Clipping",
            "SEVERITY": "Info",
            "AC_POWER_KW": round(row["AC_POWER_KW"], 2),
            "EXPECTED_AC_POWER_KW": round(row["EXPECTED_AC_POWER_KW"], 2),
            "FLEET_MEDIAN_KW": round(row["FLEET_MEDIAN_AC_KW"], 2),
            "IRRADIATION": round(row["IRRADIATION"], 3),
            "DESCRIPTION": f"AC power saturated at {row['AC_POWER_KW']:.1f} kW near nameplate threshold ({meta.nameplate_ac_kw_per_inverter:.0f} kW).",
        })

    # 4. Low Conversion Efficiency
    eff_mask = (
        (df["IRRADIATION"] >= 0.40) &
        (df["DC_POWER_KW"] >= 200.0) &
        (df["INVERTER_EFFICIENCY_PCT"] < 88.0)
    )
    for _, row in df[eff_mask].iterrows():
        anomalies.append({
            "DATE_TIME": row["DATE_TIME"],
            "DATE": row["DATE"],
            "PLANT_ID": row["PLANT_ID"],
            "SOURCE_KEY": row["SOURCE_KEY"],
            "ANOMALY_TYPE": "Low Inverter Efficiency",
            "SEVERITY": "Warning",
            "AC_POWER_KW": round(row["AC_POWER_KW"], 2),
            "EXPECTED_AC_POWER_KW": round(row["EXPECTED_AC_POWER_KW"], 2),
            "FLEET_MEDIAN_KW": round(row["FLEET_MEDIAN_AC_KW"], 2),
            "IRRADIATION": round(row["IRRADIATION"], 3),
            "DESCRIPTION": f"Inverter efficiency dropped to {row['INVERTER_EFFICIENCY_PCT']:.1f}% under high DC power ({row['DC_POWER_KW']:.1f} kW).",
        })

    anom_df = pd.DataFrame(anomalies)
    if anom_df.empty:
        return pd.DataFrame(columns=[
            "DATE_TIME", "DATE", "PLANT_ID", "SOURCE_KEY", "ANOMALY_TYPE",
            "SEVERITY", "AC_POWER_KW", "EXPECTED_AC_POWER_KW", "FLEET_MEDIAN_KW",
            "IRRADIATION", "DESCRIPTION"
        ])

    return anom_df.sort_values("DATE_TIME").reset_index(drop=True)


def summarize_anomalies_by_inverter(anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes anomaly counts and severity per inverter."""
    if anomalies_df.empty:
        return pd.DataFrame()

    summary = anomalies_df.groupby("SOURCE_KEY").agg(
        TOTAL_ANOMALIES=("ANOMALY_TYPE", "count"),
        CRITICAL_OUTAGES=("SEVERITY", lambda x: (x == "Critical").sum()),
        UNDERPERF_WARNINGS=("SEVERITY", lambda x: (x == "Warning").sum()),
        CLIPPING_EVENTS=("SEVERITY", lambda x: (x == "Info").sum()),
    ).reset_index()

    return summary.sort_values("TOTAL_ANOMALIES", ascending=False).reset_index(drop=True)
