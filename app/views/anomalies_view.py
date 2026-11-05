"""Asset Health and Anomaly Identification View for SolarScope."""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import PLANTS
from src.analytics.anomaly import summarize_anomalies_by_inverter
from app.components.plots import plot_anomaly_timeline, THEME_LAYOUT, COLOR_DANGER, COLOR_PRIMARY, COLOR_SECONDARY


def render_anomalies(plant_id: int, anomalies_df: pd.DataFrame) -> None:
    """Renders anomaly detection dashboard, event log, and inverter asset health diagnosis."""
    meta = PLANTS[plant_id]

    st.markdown(f"### ⚠️ Anomaly Detection & Asset Health: {meta.name}")
    st.markdown(
        "Automated supervisory surveillance detecting **inverter trips/outages**, "
        "**string-level underperformance**, **power clipping**, and **conversion degradation**."
    )

    if anomalies_df.empty:
        st.success("No anomalies detected in the selected plant dataset.")
        return

    # Counts by severity
    critical_count = (anomalies_df["SEVERITY"] == "Critical").sum()
    warning_count = (anomalies_df["SEVERITY"] == "Warning").sum()
    info_count = (anomalies_df["SEVERITY"] == "Info").sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events Detected", len(anomalies_df))
    with col2:
        st.metric("Critical Outages / Trips", critical_count)
    with col3:
        st.metric("Underperformance Alerts", warning_count)
    with col4:
        st.metric("Clipping / Saturated Intervals", info_count)

    st.markdown("---")

    # Interactive timeline
    st.plotly_chart(plot_anomaly_timeline(anomalies_df), use_container_width=True)

    # Inverter Anomaly Breakdown
    inv_summary = summarize_anomalies_by_inverter(anomalies_df)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("#### 🔍 Filter Operational Event Log")
        severities = ["All"] + list(anomalies_df["SEVERITY"].unique())
        selected_sev = st.selectbox("Filter by Severity:", severities, index=0)

        inverters = ["All"] + sorted(list(anomalies_df["SOURCE_KEY"].unique()))
        selected_inv = st.selectbox("Filter by Inverter ID:", inverters, index=0)

        filtered_anom = anomalies_df.copy()
        if selected_sev != "All":
            filtered_anom = filtered_anom[filtered_anom["SEVERITY"] == selected_sev]
        if selected_inv != "All":
            filtered_anom = filtered_anom[filtered_anom["SOURCE_KEY"] == selected_inv]

        st.dataframe(
            filtered_anom[[
                "DATE_TIME", "SOURCE_KEY", "ANOMALY_TYPE", "SEVERITY",
                "AC_POWER_KW", "EXPECTED_AC_POWER_KW", "IRRADIATION", "DESCRIPTION"
            ]].rename(columns={
                "DATE_TIME": "Timestamp",
                "SOURCE_KEY": "Inverter",
                "ANOMALY_TYPE": "Category",
                "SEVERITY": "Severity",
                "AC_POWER_KW": "Actual (kW)",
                "EXPECTED_AC_POWER_KW": "Expected (kW)",
                "IRRADIATION": "Irrad (kW/m²)",
                "DESCRIPTION": "Diagnosis"
            }),
            use_container_width=True,
            hide_index=True,
            height=380
        )

    with col_right:
        st.markdown("#### 🏆 Recurring Anomaly Hotspots")
        st.markdown("Top inverters generating the highest number of supervisory alerts:")
        st.dataframe(
            inv_summary.rename(columns={
                "SOURCE_KEY": "Inverter ID",
                "TOTAL_ANOMALIES": "Total",
                "CRITICAL_OUTAGES": "Outages",
                "UNDERPERF_WARNINGS": "Underperf",
                "CLIPPING_EVENTS": "Clipping"
            }),
            use_container_width=True,
            hide_index=True,
            height=380
        )
