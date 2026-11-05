"""Unit tests for SolarScope anomaly detection module."""

import pytest
import pandas as pd
from src.config import PLANTS
from src.analytics.anomaly import detect_operational_anomalies, summarize_anomalies_by_inverter


def test_detect_outage_anomaly():
    meta = PLANTS[4135001]
    # Synthetic inverter dataframe with an outage (high sun, 0 power)
    df = pd.DataFrame({
        "DATE_TIME": ["2020-05-15 12:00:00", "2020-05-15 12:00:00"],
        "DATE": ["2020-05-15", "2020-05-15"],
        "PLANT_ID": [4135001, 4135001],
        "SOURCE_KEY": ["INV_HEALTHY", "INV_TRIPPED"],
        "DC_POWER_KW": [1200.0, 0.0],
        "AC_POWER_KW": [1170.0, 0.0],  # INV_TRIPPED is down!
        "EXPECTED_AC_POWER_KW": [1200.0, 1200.0],
        "IRRADIATION": [0.85, 0.85],
        "INVERTER_EFFICIENCY_PCT": [97.5, 0.0],
    })

    anomalies = detect_operational_anomalies(df, meta)
    assert not anomalies.empty
    outages = anomalies[anomalies["ANOMALY_TYPE"] == "Inverter Outage / Trip"]
    assert len(outages) == 1
    assert outages["SOURCE_KEY"].iloc[0] == "INV_TRIPPED"
    assert outages["SEVERITY"].iloc[0] == "Critical"


def test_summarize_anomalies_by_inverter():
    anom_df = pd.DataFrame({
        "SOURCE_KEY": ["INV_01", "INV_01", "INV_02"],
        "ANOMALY_TYPE": ["Inverter Outage / Trip", "Low Inverter Efficiency", "Inverter Outage / Trip"],
        "SEVERITY": ["Critical", "Warning", "Critical"],
    })
    summary = summarize_anomalies_by_inverter(anom_df)
    assert len(summary) == 2
    assert summary.loc[summary["SOURCE_KEY"] == "INV_01", "TOTAL_ANOMALIES"].iloc[0] == 2
    assert summary.loc[summary["SOURCE_KEY"] == "INV_01", "CRITICAL_OUTAGES"].iloc[0] == 1
