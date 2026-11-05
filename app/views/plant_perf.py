"""Inverter Fleet Benchmarking and Plant Performance View for SolarScope."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import PLANTS
from src.analytics.plant_perf import benchmark_inverters, calculate_actual_vs_expected
from app.components.plots import plot_inverter_rankings, THEME_LAYOUT, COLOR_PRIMARY, COLOR_SECONDARY


def render_plant_perf(plant_id: int, inverter_df: pd.DataFrame) -> None:
    """Renders inverter-level benchmarking, health rankings, and actual vs expected models."""
    meta = PLANTS[plant_id]
    bench_df = benchmark_inverters(inverter_df, meta)
    act_exp_df = calculate_actual_vs_expected(inverter_df)

    st.markdown(f"### 🔌 Inverter Fleet Benchmarking: {meta.name}")
    st.markdown(
        f"Granular performance evaluation across all **{meta.inverter_count} inverters**. "
        "Enables asset managers to isolate sub-string failures, inverter derating, and localized array underperformance."
    )

    # Health category summary
    health_counts = bench_df["HEALTH_STATUS"].value_counts().to_dict()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("High Performers (>102%)", health_counts.get("High Performer", 0))
    with c2:
        st.metric("Normal Operating (90-102%)", health_counts.get("Normal", 0))
    with c3:
        st.metric("Underperforming (75-90%)", health_counts.get("Underperforming", 0))
    with c4:
        st.metric("Critical (<75%)", health_counts.get("Critical Underperformance", 0))

    st.markdown("---")

    # Inverter Ranking Bar Chart
    st.plotly_chart(plot_inverter_rankings(bench_df), use_container_width=True)

    # Tabs for detailed views
    tab1, tab2, tab3 = st.tabs(["Fleet Ranking Table", "Actual vs Expected Power", "Fleet Generation Heatmap"])

    with tab1:
        st.markdown("#### 📋 Inverter Health & Operational Metrics")
        st.dataframe(
            bench_df[[
                "SOURCE_KEY", "HEALTH_STATUS", "TOTAL_ENERGY_KWH",
                "SPECIFIC_YIELD_KWH_KWP", "RELATIVE_ENERGY_INDEX",
                "PEAK_AC_POWER_KW", "MEDIAN_EFFICIENCY_PCT", "MEDIAN_PR"
            ]].rename(columns={
                "SOURCE_KEY": "Inverter ID",
                "HEALTH_STATUS": "Status",
                "TOTAL_ENERGY_KWH": "Energy (kWh)",
                "SPECIFIC_YIELD_KWH_KWP": "Yield (kWh/kWp)",
                "RELATIVE_ENERGY_INDEX": "Index (vs Median)",
                "PEAK_AC_POWER_KW": "Peak AC (kW)",
                "MEDIAN_EFFICIENCY_PCT": "Efficiency (%)",
                "MEDIAN_PR": "PR"
            }).round(2),
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.markdown("#### ⚖️ Actual vs STC-Derated Expected Power")
        st.markdown(
            "Comparing aggregated fleet AC generation with theoretical STC-derated expected power "
            "accounting for instantaneous in-plane irradiance and temperature loss coefficients."
        )

        fig_scatter = px.scatter(
            act_exp_df.sample(min(2000, len(act_exp_df)), random_state=42),
            x="EXPECTED_AC_KW",
            y="ACTUAL_AC_KW",
            color="IRRADIATION",
            color_continuous_scale="Turbo",
            labels={
                "EXPECTED_AC_KW": "Theoretical Expected AC Power (kW)",
                "ACTUAL_AC_KW": "Actual AC Power (kW)",
                "IRRADIATION": "Irradiation (kW/m²)"
            },
            title="Fleet Actual vs Expected AC Power Scatter"
        )
        # Add 1:1 perfect parity line
        max_val = max(act_exp_df["EXPECTED_AC_KW"].max(), act_exp_df["ACTUAL_AC_KW"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="1:1 Parity Line"
        ))
        fig_scatter.update_layout(**THEME_LAYOUT, height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.markdown("#### 🌡️ Inverter Daily Generation Heatmap")
        st.markdown("Heatmap illustrating daily energy generation (kWh) for every inverter across the monitoring window.")

        pivot_df = inverter_df.groupby(["SOURCE_KEY", "DATE"])["INTERVAL_ENERGY_KWH"].sum().reset_index()
        heatmap_matrix = pivot_df.pivot(index="SOURCE_KEY", columns="DATE", values="INTERVAL_ENERGY_KWH")

        fig_heat = px.imshow(
            heatmap_matrix,
            labels=dict(x="Date", y="Inverter Key", color="Daily kWh"),
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Daily Energy Generation Heatmap Across Inverter Fleet"
        )
        fig_heat.update_layout(**THEME_LAYOUT, height=520)
        st.plotly_chart(fig_heat, use_container_width=True)
