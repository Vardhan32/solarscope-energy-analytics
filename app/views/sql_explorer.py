"""Interactive SQL Analytics Console for SolarScope."""

import time
import streamlit as st
import pandas as pd

from src.data.database import query_custom_sql

SAMPLE_QUERIES = {
    "1. Fleet Daily Generation & Capacity Factor Summary": """
SELECT 
    p.name as plant_name,
    s.date,
    ROUND(s.total_ac_energy_kwh / 1000.0, 2) as energy_mwh,
    ROUND(s.peak_ac_power_kw / 1000.0, 2) as peak_ac_mw,
    ROUND(s.capacity_factor_pct, 2) as capacity_factor_pct,
    ROUND(s.performance_ratio, 3) as performance_ratio,
    ROUND(s.avg_irradiation, 3) as avg_irradiance_kw_m2
FROM plant_daily_summary s
JOIN plants p ON s.plant_id = p.plant_id
ORDER BY s.plant_id, s.date DESC
LIMIT 20;
""",
    "2. Top 10 Most Productive Inverters Across Fleet": """
SELECT 
    p.name as plant_name,
    g.source_key as inverter_id,
    ROUND(SUM(g.interval_energy_kwh) / 1000.0, 2) as total_mwh,
    ROUND(MAX(g.ac_power_kw), 1) as peak_kw,
    ROUND(AVG(g.inverter_efficiency_pct), 2) as avg_efficiency_pct
FROM generation_15min g
JOIN plants p ON g.plant_id = p.plant_id
WHERE g.is_daylight = 1
GROUP BY g.plant_id, g.source_key
ORDER BY total_mwh DESC
LIMIT 10;
""",
    "3. Critical Daylight Outage Events (Irradiance > 0.25 kW/m² & AC < 5 kW)": """
SELECT 
    date_time,
    plant_id,
    source_key as inverter_id,
    anomaly_type,
    severity,
    ac_power_kw,
    expected_ac_power_kw,
    irradiation,
    description
FROM anomalies
WHERE severity = 'Critical'
ORDER BY date_time DESC
LIMIT 25;
""",
    "4. Midday Peak Hour Performance (12:00 - 13:00) Grouped by Date": """
SELECT 
    date,
    ROUND(AVG(ac_power_kw), 1) as avg_midday_ac_kw,
    ROUND(AVG(irradiation), 3) as avg_midday_irrad,
    ROUND(AVG(module_temperature), 1) as avg_module_temp_c,
    ROUND(AVG(inverter_efficiency_pct), 2) as avg_eff_pct
FROM generation_15min
WHERE plant_id = 4135001 AND hour = 12
GROUP BY date
ORDER BY date ASC;
"""
}


def render_sql_explorer() -> None:
    """Renders interactive SQL analytical query console."""
    st.markdown("### 🗄️ SQL Energy Analytics Console")
    st.markdown(
        "Execute relational SQL queries directly against the embedded SQLite database (`solarscope.db`). "
        "Demonstrates production energy data querying across inverter intervals, plant daily aggregations, and asset anomalies."
    )

    preset_name = st.selectbox("Select a Pre-Built Analytical Query:", list(SAMPLE_QUERIES.keys()))
    default_sql = SAMPLE_QUERIES[preset_name]

    query_text = st.text_area(
        "SQL Query Editor (Read-Only Analytical Queries):",
        value=default_sql.strip(),
        height=180
    )

    if st.button("▶️ Execute Query", type="primary"):
        # Security sanity check for read-only analytical explorer
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        if any(f in query_text.upper().split() for f in forbidden):
            st.error("Only read-only SELECT queries are permitted in the analytics console.")
            return

        try:
            start_time = time.perf_counter()
            result_df = query_custom_sql(query_text)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            st.success(f"Query returned **{len(result_df)} rows** in **{elapsed_ms:.1f} ms**")
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            # CSV Download
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Query Result to CSV",
                data=csv,
                file_name="solarscope_query_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")
